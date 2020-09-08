# # Deferred Revenue Forecast
#
#
# Steps to the program
# 1. Load up all input data
#     - billings history
#         - Type A
#     - FX rates
#     - FX_currency map
#     - FX forwards
#     - bookings data
#
#  Loading All of the other information we need here from excel files
#  - currency_map: contain a mapping of currency the majority of our billings in each country
#  - FX_data: contains current spot rates, FX forward rates and FX volatilities
#  - FX_forward_rates: contains the forward rates used in the FP&A Plan
#  - Bookings Forecast: contains the most recent FP&A net new booking forecast (usually only one fiscal year included)

#
# 2. Process the billings data into a dataframe that includes the BU, currency, period and every type of billings based on it's rebill frequency
#
# 3. Process the bookings information
#
# 4. Forecast the future billings
#
# 5. Basic reporting documents
#
# 6. Checking for sanity
#

""" Import Functions """
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from math import ceil
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d, griddata
import json
from deferred_revenue_functions import *

from src.deferred_revenue_functions import classify_no_POB

plt.style.use("ggplot")

# Step 1: Processing Base Billings Data
""" Data File Names and sheet names are contained in the config.json file """
with open('base_config.json') as json_file:
    config_dict1 = json.load(json_file)

with open('period_config.json') as json_file:
    config_dict2 = json.load(json_file)

# Merging the two config dictionaries
config_dict = {**config_dict1, **config_dict2}

""" Loading up the input files """
# Adobe Financial Calendar to get period start and end dates
df_cal = load_ADBE_cal(config_dict)

# FX Rates
df_FX_rates = load_FX_data(config_dict)

# FX forwards
df_FX_fwds = load_FX_fwds(config_dict)

#  Currency Map
df_curr_map = load_curr_map(config_dict)

# Base Billings File (not type A)
df, model_dict, df_no_POB, df_a_no_config, gb_d_no_rebill = load_base_billings(config_dict)

# save temporary files to a dictionary
base_bill_dict = {'df': df,
                  'model_dict': model_dict,
                  'df_no_POB': df_no_POB,
                  'df_a_no_config': df_a_no_config,
                  'gb_d_no_rebill': gb_d_no_rebill}

int_output_1_fname = config_dict['output_dir']['intermediate'] + "int_output_1.p"

with open(int_output_1_fname, "wb") as f:
    pickle.dump(base_bill_dict, f)


# Classifying the billings with no POB type
df_no_POB, gb_a_no_config_2, df_d_no_rebill = classify_no_POB(config_dict, df_no_POB)

# combine df_no_POB with df
df = pd.concat([df, df_no_POB])
df = df.groupby(["curr", "BU", "period"], as_index=False).sum()

# combine the type_A no config
df_A_no_config = pd.concat([df_a_no_config, gb_a_no_config_2])
df_A_no_config = df_A_no_config.groupby(["curr", "BU", "period"], as_index=False).sum()

# save the intermediate data to a file
post_POB_dict = {'df_no_POB': df_no_POB,
                 'gb_a_no_config_2': gb_a_no_config_2,
                 'df_d_no_rebill': df_d_no_rebill,
                 'df': df,
                 'df_A_no_config': df_A_no_config}

int_output_2_fname = config_dict['output_dir']['intermediate'] + "int_output_2.p"
with open(int_output_2_fname, "wb") as f:
    pickle.dump(post_POB_dict, f)

# process the type_A billings with no config type
df_billings = add_type_A_billings(config_dict, df, model_dict)

error_duplicates = test_df_duplicates(df_billings)
if error_duplicates:
    print("Duplicates in our billings dataframe")

df_billings = df_billings.sort_values(
    ["curr", "BU", "period"], ascending=(True, True, True))

int_output_3_fname =  config_dict['output_dir']['intermediate'] + "int_output_3.p"
with open(int_output_3_fname, "wb") as f:
    pickle.dump(df_billings, f)


# Adding periods weeks (from the Adobe calendar) to the billings dataframe
# Merging the calendar periods with the periods in the df_billings dataframe to bring over period weeks
df_billings = df_billings.merge(
    df_cal, how="left", left_on="period", right_on="period_match"
)

df_billings.drop(["period_match"], axis=1, inplace=True)

# ##### Saving these dataframes in as a python dictionary in the pickle file 'all_inputs.p'

df_billings = df_billings.sort_values(
    ["curr", "BU", "period"], ascending=(True, True, True)
)

# Bookings Forecast
df_bookings = load_bookings(config_dict)

# Merging the bookings country data to a currency using the currency map dataframe (df_curr_map)
df_bookings = merge_bookings_with_curr(df_bookings, df_curr_map)

# TODO: Check on bookings by BU and Quarter to see if it Matches Karen's file
# This needs to go into a seperate function that will dump into excel to test

# saving another intermediate file here
input_df_dict = {
    "model_dict": model_dict,
    "billings": df_billings,
    "ADBE_cal": df_cal,
    "bookings": df_bookings,
    "FX_forwards": df_FX_fwds,
    "FX_rates": df_FX_rates,
}
int_output_4_fname = config_dict['output_dir']['intermediate'] + "int_output_4.p"
with open(int_output_4_fname, "wb") as f:
    pickle.dump(input_df_dict, f)

# Cleaning up the billings dataframe
# - the billings dataframe does not contain every period if there are no bookings within a period.
# - the easiest way to create the forecast requires that we have all of the periods in each BU and currency pair (or at least 36 months worth so that we can incorporate the 3 year deferred bookings
#
#  The bookings foreacast also contains products such as 'LiveCycle' and 'other solutions' that we do not expect to recieve billings for going forward (there are no booking associated with this) so we need to remove them from the billings data
# Explicit call to the add_billings_periods function is below
print(
    "Length of df_billings before removal of old BUs and adding periods:",
    len(df_billings),
)

df_billings = add_billings_periods(df_billings)
print(
    "Length of df_billings after removal of old BUs and adding periods:",
    len(df_billings),
)

# ## Cleaning up the bookings dataframe to be incorporated into the deferred model
# - The billings dataframe is by period
# - the bookings dataframe contains net new bookings by quarter
#
df_book_period = build_booking_periods(df_bookings, df_billings)

# ##### Converting these billings to local currency based on the forward rates at the time the plan was created
# The booking forecast is in USD. I need to map this back into Document currency to forecast the billings in
# document currency. The 'plan' forward rates should have been used to create the initial USD amounts,
# so this is consistent with how we build the plan and bookings forecast
df_book_period = convert_bookings_to_DC(df_book_period, df_FX_fwds)

"""Building the billings forecast in a dataframe called df_fcst
#
# ###  Forecasting the billings into the future
# #### Steps
#  - create list of bill periods that is sorted for the lookup functions
#  - create forecast dataframe that includes the same columns (though in document currency) for the billings
#  - add the bookings forecast to this data
#  - create impact on deferred (project the new waterfall from this_
#  - load up accounting's version of the initial waterfall (by BU)
#  - reporting
"""
df_fcst = configure_df_fcst(df_billings, df_cal, config_dict)
'''
# creating the list of historical bill periods
v_BU = df_billings["BU"].copy()
v_curr = df_billings["curr"].copy()
v_both = v_BU + v_curr
v_unique = v_both.unique()

v_un_BU = [sub[:-3] for sub in v_unique]
v_un_curr = [sub[-3:] for sub in v_unique]

list_future_periods = config_dict['list_future_periods']

# creating the vectors for the future billings dataframe
v_BU_2_df = []
v_curr_2_df = []
v_period_2_df = []

for i in range(len(v_un_BU)):
    this_BU = v_un_BU[i]
    this_curr = v_un_curr[i]

    for period in list_future_periods:
        v_BU_2_df.append(this_BU)
        v_curr_2_df.append(this_curr)
        v_period_2_df.append(period)

print("This is the length of the vectors: ", len(v_BU_2_df))

# ##### Creating a list of the columns that we need to use in the df_billings dataframe (They contain document currency billings)

list_all_columns = df_billings.columns

list_keepers = []
for i in list_all_columns:

    if i[-2:] == "DC":
        list_keepers.append(i)

# ##### Creating the df_fcst dataframe with every currency, BU and period we need
df_fcst = pd.DataFrame({"curr": v_curr_2_df, "BU": v_BU_2_df, "period": v_period_2_df})

#  Adding the columns we need to populate (list_keepers)

for col in list_keepers:
    df_fcst[col] = 0

df_fcst = df_fcst.merge(df_cal, how="left", left_on="period", right_on="period_match")
df_fcst.drop(["period_match"], axis=1, inplace=True)
'''
# ### The functions below create the billings forecast by looking up the historical billings and having them renew
# NOTE: The monthly billings are using a linear regression model on the monthly billings / weeks in the month

#
# The monthly time series is used to create a best fit linear refgression model. The entire dataset is unsed in the initial model. Earlier periods get dropped and the model rerun to determine if a shorter time horizon increases the fit (R-squared) of the model. The best fitted model is then used to forecast the monthly billings.
#
# The r-squared, coefficient, intercept and number of periods from the best fit model are stored in the df_fcst dataframe
# ### TO BE DONE TO COMPLETE:
#
#  - determine which print statement need to be kept to make sure it is running appropriately
#  - remove slice error warnings after investigating where the problem occurs

df_fcst = create_billing_forecast(df_billings, df_fcst)

#test_output = df_fcst[(df_fcst["curr"] == "EUR") & (df_fcst["BU"] == "Creative")]
#test_output.head(20)

# ### saving the initial work here

df_billings = df_billings.sort_values(
    ["curr", "BU", "period"], ascending=(True, True, True)
)
df_fcst = df_fcst.sort_values(["curr", "BU", "period"], ascending=(True, True, True))

input_df_dict = {
    "model_dict": model_dict,
    "billings": df_billings,
    "ADBE_cal": df_cal,
    "bookings": df_book_period,
    "FX_forwards": df_FX_fwds,
    "FX_rates": df_FX_rates,
    "forecast": df_fcst,
}
int_output_5_fname = config_dict['output_dir']['intermediate'] + "int_output_5.p"
with open(int_output_5_fname, "wb") as f:
    pickle.dump(input_df_dict, f)

# ##### Creating USD amounts for the forecast
# We have the document currency forecast of billings and now have to translate this into USD
#  - df_fcst (contains the forecast and historical billings)
#  - FX_rates (contains current spot rates, forward rates and volatilities)
#
#  ##### I will need to create a 12 month forward vector for each currency
#   - First add 'is_direct' field to the df_FX_rates DataFrame

df_FX_rates = interp_FX_fwds(df_FX_rates)


# #### Creating USD forecast
#  - create the loop to iterate over each BU and currency (function is already written and used above)
#  - multiply each of the following by the tranp_fwds to get the USD amounts
#      - recognized_DC
#      - service_DC
#      - deferred_1M_DC
#      - deferred_3M_DC
#      - deferred_6M_DC
#      - deferred_1Y_DC
#      - deferred_2Y_DC
#      - deferred_3Y_DC
#
#  - make this function easy to use by tweaking the forward rates or shocking the FX_rates
#       - This will be a call to the df_FX_rates which will then recalculate the forward rates
#       - Then the new forward rates will need to be fed back into the df_fcst dataframe to recalculate USD amounts

new_columns = config_dict["new_columns"]
list_columns = config_dict["list_columns"]

df_fcst = convert_fcst(df_fcst, df_FX_rates, list_columns, new_columns)

# ###### Checking with a slice of the dataframe df_fcst
#us_slice = df_fcst[(df_fcst["BU"] == "Creative") & (df_fcst["curr"] == "JPY")]
#us_slice.head(10)

#dc = us_slice["deferred_1Y_DC"]
#us = us_slice["deferred_1Y_US"]
#print(dc / us)

# ### Adding the bookings data to the df_fcst. columns

# #### Need to take each BU/curr combination in the df_book_period rows and pull
# P07, P08, P09 ... P12 and P07_US, P08_US ... P12_US and move them to the df_fcst
# dataframe under the correct BU / curr / period section
df_fcst = merge_bookings_to_fcst(df_book_period, df_fcst)

print(df_fcst.columns)

# Creating the deferred revenue waterfall from our forecasted billings dataframe
df_wf = build_deferred_waterfall(df_fcst)

df_wf = df_wf.reset_index(drop=True)

# The billings file is at the Enterprise BU level and the bookings forecast is at the BU level

# ### Change the BU to match the waterfall BUs
#  - Creative to Digital Media
#  - Document Cloud to Digital Media
#  - Print & Publishing to Publishing
#  - DX Other to Digital Experience
#  - Experience Cloud to Digital Experience

dict_BU_map = config_dict['bookings_BU_mapping']
for key, value in dict_BU_map.items():
    df_wf["BU"] = df_wf["BU"].str.replace(key, value)

#df_wf["BU"] = df_wf["BU"].str.replace("Creative", "Digital Media")
#df_wf["BU"] = df_wf["BU"].str.replace("Document Cloud", "Digital Media")
#df_wf["BU"] = df_wf["BU"].str.replace("DX Other", "Digital Experience")
#df_wf["BU"] = df_wf["BU"].str.replace("Experience Cloud", "Digital Experience")
#df_wf["BU"] = df_wf["BU"].str.replace("Print & Publishing", "Publishing")

# The df_wf contains USD equivalent amounts but is still in DC groupings
# # I need to groupby BU and Period

df_wf = df_wf.groupby(["BU", "period"]).sum()

# ### Now we need to group the BU to match the waterfall BUs
df_wf.reset_index(inplace=True)

# Loading up the initial accounting deferred revenue waterfall
df_wf_init = load_and_clean_init_waterfall(config_dict)

# ## Take the As Performed / Upon Acceptance column and place this into the df_wf_gb dataframe.
# ## We will assume that this does not change over time
df_as_performed = df_wf_init["As Performed / Upon Acceptance"].copy()
df_as_performed = df_as_performed.to_frame()
df_wf_init = df_wf_init.drop("As Performed / Upon Acceptance", axis=1)

# Changing the periods in the df_wf_gb to match the df_wf_init first
old_cols = df_wf.columns
old_cols = old_cols[3:]

new_columns = []
for i in range(12 * 3):
    if len(str(i + 1)) == 1:
        new_column = "P0" + str(i + 1)
    else:
        new_column = "P" + str(i + 1)
    new_columns.append(new_column)

rename_dict = dict(zip(old_cols, new_columns))
df_wf = df_wf.rename(columns=rename_dict)

list_periods = df_wf.period.unique()

list_BU = df_wf.BU.unique()

df_wf_init["period"] = "2020-10"

df_wf_init = df_wf_init.reset_index()

df_wf_init.rename(columns={"External Reporting BU": "BU"}, inplace=True)

# ## ADDING ADDITIONAL PERIODS HERE TO MERGE WITH df_wf

df_wf_init["P28"] = 0
df_wf_init["P29"] = 0
df_wf_init["P30"] = 0
df_wf_init["P31"] = 0
df_wf_init["P32"] = 0
df_wf_init["P33"] = 0
df_wf_init["P34"] = 0
df_wf_init["P35"] = 0
df_wf_init["P36"] = 0

# Planning on making a new dataframe with BU and period and merging them.
# Will create NAs everywhere else, but we will fillna
list_periods = list_periods[1:]

to_df_BU = []
to_df_period = []
for item in list_BU:

    for per in list_periods:
        to_df_BU.append(item)
        to_df_period.append(per)

len(to_df_BU)

# Creating new dataframe to be merged
df_to_merge = pd.DataFrame({"BU": to_df_BU, "period": to_df_period})

df_wf_init = df_wf_init.merge(df_to_merge, on=["BU", "period"], how="outer")

df_wf_init.fillna(0, inplace=True)

df_wf_init.sort_values(by=["BU", "period"], inplace=True)

# ## Now move the waterfall forward by BU
df_wf_init = bring_initial_wf_forward(df_wf_init)

df_wf_init = df_wf_init.set_index(["BU", "period"])

df_wf = df_wf.set_index(["BU", "period"])

df_all = df_wf_init.add(df_wf, fill_value=0)

df_wf_init = df_wf_init.sort_index()
df_all = df_all.sort_index()
df_wf = df_wf.sort_index()

# #### Sending this data over to excel as a check
with pd.ExcelWriter("output.xlsx") as writer:
    df_wf_init.to_excel(writer, sheet_name="initial_waterfall")
    df_wf.to_excel(writer, sheet_name="billings_impact")
    df_all.to_excel(writer, sheet_name="combined")
    df_as_performed.to_excel(writer, sheet_name="as_performed")
    #df_wf.to_excel(writer, sheet_name="early_wf")


# ## Add the as performed back into the waterfall forecast

df_all["Total"] = df_all[df_all.columns[:-1]].sum(axis=1)

df_wf_init["Total"] = df_wf_init[df_wf_init.columns[:]].sum(axis=1)

df_wf["Total"] = df_wf[df_wf.columns[1:]].sum(axis=1)

# # Need to add these back to our model dictionary
# ## At this point I dont recall what it was called and will need to change it

saved_dict = {}
saved_dict["waterfall"] = df_all
saved_dict["bill_waterfall"] = df_wf
saved_dict["initial_waterfall"] = df_wf_init

pickle.dump(saved_dict, open("../data/processed/final_forecast_2.p", "wb"))

# ### Testing parts of the bookings/waterfall
#list_Q4 = ["2020-10", "2020-11", "2020-12"]
#this_BU = "Creative"
#test_Q4 = df[(df["BU"] == this_BU) & (df["period"].isin(list_Q4))]

#test_Q4["book_1Y_US"].sum()

# Merging the df_fcst with the df_bililngs for easier charting?

df_billings["is_forecast"] = 0
df_fcst["is_forecast"] = 1
df = pd.concat([df_billings, df_fcst], join="outer", ignore_index=True)
df = df.fillna(0)
df.sort_values(by=["curr", "BU", "period"], inplace=True)

input_df_dict = {
    "model_dict": model_dict,
    "ADBE_cal": df_cal,
    "bookings": df_bookings,
    "FX_forwards": df_FX_fwds,
    "FX_rates": df_FX_rates,
    "final": df,
    "billings": df_billings,
    "forecast": df_fcst,
    "initial_waterfall": df_wf_init,
    "bill_waterfall": df_wf,
    "waterfall": df_all,
    "as_performed": df_as_performed
}

pickle.dump(input_df_dict, open("../data/processed/final_forecast3.p", "wb"))


