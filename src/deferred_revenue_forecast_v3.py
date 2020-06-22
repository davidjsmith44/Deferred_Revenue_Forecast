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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")

import pickle
from math import ceil
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d, griddata
import logging
import deferred_revenue_functions

logger = logging.getLogger("deferred_logger")
logger.setLevel("DEBUG")
logging.basicConfig(
    filename="example.log",
    level=logging.DEBUG,
    format="[%(filename)s:%(lineno)s - %(funcName)15s()] %(message)s",
)


# ## Step 1: Processing Base Billings Data
logger.debug("Processing the Base Billings Data")

billings_filename = "../data/Data_2020_P06/all_billings_inputs.xlsx"
billings_sheetname = 'base_billings'
type_A_sheetname = "type_A_no_config"

ADBE_cal_filename = "../data/old/ADOBE_FINANCIAL_CALENDAR.xlsx"
ADBE_cal_sheetname = 'ADBE_cal'
curr_map_filename = "../data/Data_2020_P06/currency_map.xlsx"
curr_map_sheetname ="curr_map"

# ##### FX data
FX_rates_filename = "../data/Data_2020_P06/FX_data.xlsx"
FX_rates_sheetname = 'to_matlab'
df_FX_rates = load_FX_data(FX_rates_filename, FX_rates_sheetname)

# ###### FX Forward Rates used in the FP&A Plan
FX_fwds_filename = "../data/Data_2020_P06/FX_forward_rates.xlsx"
FX_fwds_sheetname = 'forward_data'
df_FX_fwds = load_FX_fwds(FX_fwds_filename, FX_fwds_sheetname)

# ##### Bookings Forecast
bookings_filename = "../data/Data_2020_P06/2020_bookings_fcst_Q2.xlsx"
bookings_sheetname = 'source'

# ###### Loading up the Adobe Financial Calendar to get period start and end dates
df_cal = load_ADBE_cal(ADBE_cal_filename, ADBE_cal_sheetname)

# ###### Currency Map
df_curr_map = load_curr_map(curr_map_filename, curr_map_sheetname)

df, model_dict = load_base_billings(billings_filename, billings_sheetname)


# ## I NEED TO CREATE A BETTER PRESENTATION OF THIS CHECK THAT EVERYTHING MATCHES!!!!
# ## Need to create a summary report with totals coming from every area to make sure the totals I have make sense
df.sum()

total_df = sum_USD_amt(list_df, list_columns)
total_df

total_df.loc["deferred_1M_d"] + total_df.loc["deferred_1M_a"]


# # TO BE DONE:
#
# 1. Clean up the type F billings (at least check to see if they are necessary)
#

# ___
# ## Type A No Config Type Billings
# ___
#
# This file contains type A billings that have a revenue contract start date and end date. We need to map these into the terms of our dataframe.
#
# #### Steps
# 1. Rename the columns
# 2. This file has entries for pennies. Need to clear out anything less than $10 in absolute value
# 3. Determine the length of time between start date and end date
# 4. Group this dataframe by currency, period and BU
# 5. Merge this final dataframe with the larger dataframe
#

# ###### Note: This file contains two different start date and end date columns. At least one of these columns is populated

df_billings = add_type_A_billings(billings_filename, type_A_sheetname, df, model_dict)

with open("../data/processed/all_billings.p", "wb") as f:
    pickle.dump(df_billings, f)


# #### Below Load up the billings dataframe (for debugging purposes)


# df_billings = pickle.load( open('../data/processed/all_billings.p', 'rb' ))


# ### Loading All of the other information we need here from excel files
#  - currency_map: contain a mapping of currency the majority of our billings in each country
#  - FX_data: contains current spot rates, FX forward rates and FX volatilities
#  - FX_forward_rates: contains the forward rates used in the FP&A Plan
#  - Bookings Forecast: contains the most recent FP&A net new booking forecast (usually only one fiscal year included)

# ##### Bookings Forecast
df_bookings =load_bookings(bookings_filename, bookings_sheetname)

# ### Merging the bookings country data to a currency using the currency map dataframe (df_curr_map)

list_book_ctry = df_bookings["country"].unique()
print("Countries in the bookings file: \n", list_book_ctry)

list_curr_map = df_curr_map["Country"].unique()
print("Countries in the currency map file: \n", list_curr_map)


# ##### Checking that we have the currency mapping for every country where we have a bookings forecast


a = list(set(list_book_ctry) & set(list_curr_map))

not_in_map = set(list_book_ctry).difference(set(list_curr_map))
if len(not_in_map) != 0:
    print(
        "There is a bookings currency that is not in the currency map!\nWe need to look into the currency map file and add this!"
    )
else:
    print(
        "The bookings currencies are in the currency map. OK to merge the dataframes."
    )


# ###### Merge the bookings forecast with the currency map


df_bookings = pd.merge(
    df_bookings, df_curr_map, how="left", left_on="country", right_on="Country"
)
# the country and Country are the same so we are dropping one of them
df_bookings = df_bookings.drop("Country", axis=1)


df_bookings["booking_type"].value_counts()


# ### Check on bookings by BU and Quarter to see if it Matches Karen's file


df_cr = df_bookings[
    (df_bookings["BU"] == "Creative") & (df_bookings["Quarter"] == "Q3 2020")
]
df_cr["US_amount"].sum()


df_dc = df_bookings[
    (df_bookings["BU"] == "Document Cloud") & (df_bookings["Quarter"] == "Q3 2020")
]
df_dc["US_amount"].sum()


df_de = df_bookings[
    (df_bookings["BU"] == "Experience Cloud") & (df_bookings["Quarter"] == "Q3 2020")
]
df_de["US_amount"].sum()


df_cr4 = df_bookings[
    (df_bookings["BU"] == "Creative") & (df_bookings["Quarter"] == "Q4 2020")
]
df_cr4["US_amount"].sum()


df_dc4 = df_bookings[
    (df_bookings["BU"] == "Document Cloud") & (df_bookings["Quarter"] == "Q4 2020")
]
df_dc4["US_amount"].sum()


df_de4 = df_bookings[
    (df_bookings["BU"] == "Experience Cloud") & (df_bookings["Quarter"] == "Q4 2020")
]
df_de4["US_amount"].sum()


df_bookings.BU.value_counts()


# ### Adding periods weeks (from the Adobe calendar) to the billings dataframe



# ##### Merging the calendar periods with the periods in the df_billings dataframe to bring over period weeks

df_billings = df_billings.merge(
    df_cal, how="left", left_on="period", right_on="period_match"
)


df_billings.columns

# df_billings.drop(['period_match', '_merge'], axis=1, inplace=True)
df_billings.drop(["period_match"], axis=1, inplace=True)


df_billings.columns


df_billings.head(5)
# df_billings.sample(5)
# df_billings.tail(5)


# ##### Saving these dataframes in as a python dictionary in the pickle file 'all_inputs.p'


df_billings = df_billings.sort_values(
    ["curr", "BU", "period"], ascending=(True, True, True)
)

input_df_dict = {
    "model_dict": model_dict,
    "billings": df_billings,
    "ADBE_cal": df_cal,
    "bookings": df_bookings,
    "FX_forwards": df_FX_fwds,
    "FX_rates": df_FX_rates,
}

pickle.dump(input_df_dict, open("../data/processed/all_inputs.p", "wb"))

# ### Cleaning up the billings dataframe
# - the billings dataframe does not contain every period if there are no bookings within a period.
# - the easiest way to create the forecast requires that we have all of the periods in each BU and currency pair (or at least 36 months worth so that we can incorporate the 3 year deferred bookings
#
# ###### The bookings foreacast also contains products such as 'LiveCycle' and 'other solutions' that we do not expect to recieve billings for going forward (there are no booking associated with this) so we need to remove them from the billings data


# ###### Explicit call to the add_billings_periods function is below
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


# find the last period in the billings index
last_period = "2020-06"

list_BUs = df_bookings["BU"].unique()
list_curr = df_bookings["Currency"].unique()

print("This is the list of BUs in the bookings dataframe: ", list_BUs)
print("This is the list of currencies in the bookings dataframe: ", list_curr)


# ##### Creating data to add to the billings dataframe to incorporate period by period billings
# NOTE:  This is just creating the space in the dataframe for the data. We will fill it in later


# creating dataframe of zeros
l_BU = []
l_curr = []
for BU in list_BUs:
    for curr in list_curr:
        l_BU.append(BU)
        l_curr.append(curr)
# print(l_BU)
# print(l_curr)
l_zero = np.zeros(len(l_BU))


data = {
    "BU": l_BU,
    "curr": l_curr,
    "Q1": l_zero,
    "Q2": l_zero,
    "Q3": l_zero,
    "Q4": l_zero,
    "P01": l_zero,
    "P02": l_zero,
    "P03": l_zero,
    "P04": l_zero,
    "P05": l_zero,
    "P06": l_zero,
    "P07": l_zero,
    "P08": l_zero,
    "P09": l_zero,
    "P10": l_zero,
    "P11": l_zero,
    "P12": l_zero,
}

df_book_period = pd.DataFrame(data)


df_book_period.head(14)


# ##### Uncomment below to remember what the df_bookings looked like


df_bookings.head(10)
# df_bookings.sample(10)
# df_bookings.tail(10)


df_bookings.BU.value_counts()


# ##### The cell below fills in the df_book_period dataframe with the quarterly bookings numbers for each BU and currency


# fill in the quarters
for i in range(len(df_book_period["BU"])):

    this_BU = df_book_period["BU"][i]
    this_curr = df_book_period["curr"][i]
    this_slice = df_bookings[
        (df_bookings["BU"] == this_BU) & (df_bookings["Currency"] == this_curr)
    ]

    this_Q1 = this_slice[this_slice["Quarter"] == "Q1 2020"]
    sum_Q1 = this_Q1["US_amount"].sum()
    df_book_period["Q1"].loc[i] = sum_Q1

    this_Q2 = this_slice[this_slice["Quarter"] == "Q2 2020"]
    sum_Q2 = this_Q2["US_amount"].sum()
    df_book_period["Q2"].loc[i] = sum_Q2

    this_Q3 = this_slice[this_slice["Quarter"] == "Q3 2020"]
    sum_Q3 = this_Q3["US_amount"].sum()
    df_book_period["Q3"].loc[i] = sum_Q3

    this_Q4 = this_slice[this_slice["Quarter"] == "Q4 2020"]
    sum_Q4 = this_Q4["US_amount"].sum()
    df_book_period["Q4"].loc[i] = sum_Q4


df_book_period.head(30)
# df_book_period.sample(10)
# df_book_period.tail(10)


print("Q1 total bookings ", df_book_period["Q1"].sum())
print("Q2 total bookings ", df_book_period["Q2"].sum())
print("Q3 total bookings ", df_book_period["Q3"].sum())
print("Q4 total bookings ", df_book_period["Q4"].sum())


# ##### Creating lists of periods and quarters needed to fill out the df_book_period dataframe


# list of quarters for the percentages

list_q3 = ["2019-07", "2019-08", "2019-09"]
list_q4 = ["2019-10", "2019-11", "2019-12"]
list_q1 = ["2020-01", "2020-02", "2020-03"]
list_q2 = ["2020-04", "2020-05", "2020-06"]

list_periods = [
    "2020-01",
    "2020-02",
    "2020-03",
    "2020-04",
    "2020-05",
    "2020-06",
    "2019-07",
    "2019-08",
    "2019-09",
    "2019-10",
    "2019-11",
    "2019-12",
]

list_p_headers = [
    "P01",
    "P02",
    "P03",
    "P04",
    "P05",
    "P06",
    "P07",
    "P08",
    "P09",
    "P10",
    "P11",
    "P12",
]

list_q_headers = [
    "Q1",
    "Q1",
    "Q1",
    "Q2",
    "Q2",
    "Q2",
    "Q3",
    "Q3",
    "Q3",
    "Q4",
    "Q4",
    "Q4",
]


# ##### adding the booking periods to the dataframe. The bookings are split into periods based on last years percentage of 1 year deferred billings within the quarter.
# For example: P1 = 25%, P2 = 30%, P3 = 45% such that the sum is equal to the total quarterly billings last year


for i in range(len(df_book_period["BU"])):

    this_BU = df_book_period["BU"][i]
    this_curr = df_book_period["curr"][i]

    this_slice = df_billings[
        (df_billings["BU"] == this_BU) & (df_billings["curr"] == this_curr)
    ]

    for j in range(len(list_periods)):
        this_period = list_periods[j]
        this_header = list_p_headers[j]
        this_quarter = list_q_headers[j]
        this_P_slice = this_slice[this_slice["period"] == this_period]
        df_book_period.loc[[i], [this_header]] = this_P_slice["deferred_1Y_DC"].sum()

df_book_period["bill_Q1_sum"] = (
    df_book_period["P01"] + df_book_period["P02"] + df_book_period["P03"]
)
df_book_period["bill_Q2_sum"] = (
    df_book_period["P04"] + df_book_period["P05"] + df_book_period["P06"]
)
df_book_period["bill_Q3_sum"] = (
    df_book_period["P07"] + df_book_period["P08"] + df_book_period["P09"]
)
df_book_period["bill_Q4_sum"] = (
    df_book_period["P10"] + df_book_period["P11"] + df_book_period["P12"]
)

df_book_period["P01"] = (
    df_book_period["Q1"] * df_book_period["P01"] / df_book_period["bill_Q1_sum"]
)
df_book_period["P02"] = (
    df_book_period["Q1"] * df_book_period["P02"] / df_book_period["bill_Q1_sum"]
)
df_book_period["P03"] = (
    df_book_period["Q1"] * df_book_period["P03"] / df_book_period["bill_Q1_sum"]
)

df_book_period["P04"] = (
    df_book_period["Q2"] * df_book_period["P04"] / df_book_period["bill_Q2_sum"]
)
df_book_period["P05"] = (
    df_book_period["Q2"] * df_book_period["P05"] / df_book_period["bill_Q2_sum"]
)
df_book_period["P06"] = (
    df_book_period["Q2"] * df_book_period["P06"] / df_book_period["bill_Q2_sum"]
)

df_book_period["P07"] = (
    df_book_period["Q3"] * df_book_period["P07"] / df_book_period["bill_Q3_sum"]
)
df_book_period["P08"] = (
    df_book_period["Q3"] * df_book_period["P08"] / df_book_period["bill_Q3_sum"]
)
df_book_period["P09"] = (
    df_book_period["Q3"] * df_book_period["P09"] / df_book_period["bill_Q3_sum"]
)

df_book_period["P10"] = (
    df_book_period["Q4"] * df_book_period["P10"] / df_book_period["bill_Q4_sum"]
)
df_book_period["P11"] = (
    df_book_period["Q4"] * df_book_period["P11"] / df_book_period["bill_Q4_sum"]
)
df_book_period["P12"] = (
    df_book_period["Q4"] * df_book_period["P12"] / df_book_period["bill_Q4_sum"]
)


# df_book_period.head(10)
# df_book_period.sample(10)
df_book_period.tail(10)


# ###### Cleaning up the dataframe by dropping the columns we no longer need


df_book_period.drop(
    ["bill_Q1_sum", "bill_Q2_sum", "bill_Q3_sum", "bill_Q4_sum"], axis=1, inplace=True
)


df_book_period.columns

test_ec = df_book_period[df_book_period["BU"] == "Experience Cloud"]
test_ec["Q3"].sum()


test_ec

print("p7: ", test_ec["P07"].sum())
print("p8: ", test_ec["P08"].sum())
print("p9: ", test_ec["P09"].sum())
print(
    "Total Exp Cloud Q3: ",
    test_ec["P07"].sum() + test_ec["P08"].sum() + test_ec["P09"].sum(),
)


test_ec["Q3"].sum()


# ##### Converting these billings to local currency based on the forward rates at the time the plan was created
# The booking forecast is in USD. I need to map this back into Document currency to forecast the billings in document currency. The 'plan' forward rates should have been used to create the initial USD amounts, so this is consistent with how we build the plan and bookings forecast


df_FX_fwds.set_index("curr", inplace=True)

list_fwds = []
for i in range(len(df_book_period["curr"])):
    this_curr = df_book_period["curr"][i]

    if this_curr == "USD":
        this_fwd = 1
    else:
        this_fwd = df_FX_fwds.loc[this_curr, "forward"]

    list_fwds.append(this_fwd)
df_book_period["FX_fwd_rate"] = list_fwds

df_book_period["P01_DC"] = df_book_period["P01"] * df_book_period["FX_fwd_rate"]
df_book_period["P02_DC"] = df_book_period["P02"] * df_book_period["FX_fwd_rate"]
df_book_period["P03_DC"] = df_book_period["P03"] * df_book_period["FX_fwd_rate"]
df_book_period["P04_DC"] = df_book_period["P04"] * df_book_period["FX_fwd_rate"]
df_book_period["P05_DC"] = df_book_period["P05"] * df_book_period["FX_fwd_rate"]
df_book_period["P06_DC"] = df_book_period["P06"] * df_book_period["FX_fwd_rate"]
df_book_period["P07_DC"] = df_book_period["P07"] * df_book_period["FX_fwd_rate"]
df_book_period["P08_DC"] = df_book_period["P08"] * df_book_period["FX_fwd_rate"]
df_book_period["P09_DC"] = df_book_period["P09"] * df_book_period["FX_fwd_rate"]
df_book_period["P10_DC"] = df_book_period["P10"] * df_book_period["FX_fwd_rate"]
df_book_period["P11_DC"] = df_book_period["P11"] * df_book_period["FX_fwd_rate"]
df_book_period["P12_DC"] = df_book_period["P12"] * df_book_period["FX_fwd_rate"]


# df_book_period.head(10)
# df_book_period.sample(10)
df_book_period.tail(10)


# ##### The df_book_period dataframe now has columns for bookings each period in both local currency and document currency

df_book_period.columns


# ## Building the billings forecast in a dataframe called df_fcst
#
# ###  Forecasting the billings into the future
# #### Steps
#  - create list of bill periods that is sorted for the lookup functions
#  - create forecast dataframe that includes the same columns (though in document currency) for the billings
#  - add the bookings forecast to this data
#  - create impact on deferred (project the new waterfall from this_
#  - load up accounting's version of the initial waterfall (by BU)
#  - reporting

# ###### creating the list of historical bill periods


list_bill_periods = df_billings["period"].unique()
list_bill_periods.sort()
print(list_bill_periods)

v_BU = df_billings["BU"].copy()
v_curr = df_billings["curr"].copy()
v_both = v_BU + v_curr
v_unique = v_both.unique()

v_un_BU = [sub[:-3] for sub in v_unique]
v_un_curr = [sub[-3:] for sub in v_unique]

list_future_periods = [
    "2020-07",
    "2020-08",
    "2020-09",
    "2020-10",
    "2020-11",
    "2020-12",
    "2021-01",
    "2021-02",
    "2021-03",
    "2021-04",
    "2021-05",
    "2021-06",
]


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

list_keepers


# ##### Creating the df_fcst dataframe with every currency, BU and period we need


df_fcst = pd.DataFrame({"curr": v_curr_2_df, "BU": v_BU_2_df, "period": v_period_2_df})


# ###### Adding the columns we need to populate (list_keepers)

for col in list_keepers:
    df_fcst[col] = 0


df_fcst.head(10)
# df_fcst.sample(10)
# df_fcst.head(10)




df_fcst = df_fcst.merge(
    df_cal, how="left", left_on="period", right_on="period_match"
)
df_fcst.drop(["period_match"], axis=1, inplace=True)


df_fcst.head(10)
# df_fcst.sample(10)
# df_fcst.tail(10)


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


df_fcst.head(40)
# df_fcst.sample(20)
# df_fcst.tail(20)


test_output = df_fcst[(df_fcst["curr"] == "EUR") & (df_fcst["BU"] == "Creative")]
test_output.head(20)


df_fcst.columns


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

pickle.dump(input_df_dict, open("../data/processed/initial_forecast.p", "wb"))


df_billings.head(5)


df_fcst.tail(5)


df_FX_rates


# ##### Creating USD amounts for the forecast
# We have the document currency forecast of billings and now have to translate this into USD
#  - df_fcst (contains the forecast and historical billings)
#  - FX_rates (contains current spot rates, forward rates and volatilities)
#
#  ##### I will need to create a 12 month forward vector for each currency
#   - First add 'is_direct' field to the df_FX_rates DataFrame




df_FX_rates = interp_FX_fwds(df_FX_rates)


df_FX_rates.head(15)

df_FX_rates.columns


# #### Creating USD forecast
#  - create the loop to iterate over each BU and currency (function is already written and used above)
#  - multiply each of the following by the tranp_fwds to get the USD amounts
#      - recognized_DC
#      - service_DC
#      - deferred_B_DC
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


new_columns = [
    "fwd_01M",
    "fwd_02M",
    "fwd_03M",
    "fwd_04M",
    "fwd_05M",
    "fwd_06M",
    "fwd_07M",
    "fwd_08M",
    "fwd_09M",
    "fwd_10M",
    "fwd_11M",
    "fwd_12M",
]


list_columns = [
    "recognized_",
    "service_",
    "deferred_B_",
    "deferred_1M_",
    "deferred_3M_",
    "deferred_6M_",
    "deferred_1Y_",
    "deferred_2Y_",
    "deferred_3Y_",
]


df_fcst.columns

df_fcst = convert_fcst(df_fcst, df_FX_rates, list_columns, new_columns)

df_fcst.head(40)


# ###### Checking with a slice of the dataframe df_fcst
us_slice = df_fcst[(df_fcst["BU"] == "Creative") & (df_fcst["curr"] == "JPY")]
us_slice.head(10)

dc = us_slice["deferred_1Y_DC"]
us = us_slice["deferred_1Y_US"]
print(dc / us)


df_fcst.columns


df_billings.columns


# ### Adding the bookings data to the df_fcst. columns

df_book_period.head(10)


# #### Need to take each BU/curr combination in the df_book_period rows and pull P07, P08, P09 ... P12 and P07_US, P08_US ... P12_US and move them to the df_fcst dataframe under the correct BU / curr / period section



df_fcst = merge_bookings_to_fcst(df_book_period, df_fcst)


df_fcst.head(10)

df_fcst["book_1Y_US"].sum()


test_EUR = df_fcst[df_fcst["curr"] == "EUR"]
test_EUR


# #### Loading up the deferred revenue waterfall

# ### Loading the Deferred Revenue Forecast Sheet
#
# Steps to cleaning this notebook
#
# - clear out rows below the Grand Total inclusive of Magento/Marketo
# - forward fill the External Reporting BU
# - Move Marketo and Magento BU to Digital Experience
# - Aggregate this by External reporting BU
# - rename columns without " ' "
# - create interpolated periods here for the amortization (assume amortization to revenue is linear within the periods of a quarter
#


def load_and_clean_waterfall(waterfall_filename, waterfall_sheetname)

    df = pd.read_excel(
        "../data/Data_2020_P06/Q2'20 Rev Acctg Mgmt Workbook (06-04-20).xlsx",
        sheet_name="Deferred Revenue Forecast",
        skiprows=5,
    )


    df.head(50)


    df.columns


    # ##### Stripping spaces from the External Reporting BU columns


    df["External Reporting BU"] = df["External Reporting BU"].str.strip()


    # ##### Clearing out rows below the table we need


    end_loc = df[df["External Reporting BU"] == "Grand Total inclusive of Magento/Marketo"]
    end_index = end_loc.index[0]

    df = df[df.index <= end_index]

    df["External Reporting BU"].value_counts()


    # ### We are just taking the following rows
    # - Digital Media Total
    # - Publishing Total
    # - Digital Experience Total
    # - Marketo Deferred
    # - Magento Deferred
    #
    # Then we need to add the Marketo and Magento defered to the digital experience total


    keeper_rows = [
        "Digital Media Total",
        "Publishing Total",
        "Digital Experience Total",
        "Marketo Deferred",
        "Magento Deferred",
        "Grand Total inclusive of Magento/Marketo",
    ]

    df_test = df[df["External Reporting BU"].isin(keeper_rows)]
    df_test.head(10)


    # ##### Cleaning out the bad colunns


    df_test.columns


    df_test = df_test.loc[:, ~df_test.columns.str.contains("^Unnamed")]
    df_test = df_test.drop(columns=["Major Product Config", " Historical"])


    df_test.head(10)


    # ## Add Magento and Marketo to Digital Experience
    #
    # ##### NOTE: The External Reporting BU is different the the BU we have in deferred.
    # We will have to combine Creative and Document Cloud to get to the External Reporting BU since both show up as Digital Media in this accounting workbook
    #


    df_test["External Reporting BU"] = df_test["External Reporting BU"].str.replace(
        "Magento Deferred", "Digital Experience Total"
    )

    df_test["External Reporting BU"] = df_test["External Reporting BU"].str.replace(
        "Marketo Deferred", "Digital Experience Total"
    )


    # ##### Removing non-alphanumeric characters


    changed_columns = df_test.columns.str.replace("'", "_")
    changed_columns = changed_columns.str.replace("+", "")
    df_test.columns = changed_columns


    df_test.columns


    df_test

    df_test_gb = df_test.groupby("External Reporting BU").sum()


    df_test_gb


    # ### Now that we have the data that is all numeric, we need to adjust for the reporting in thousands (FP&A report)


    df_test_gb = df_test_gb * 1000


    df_test_gb


    # ### Creating the columns that have this amortization by period
    #
    # #### Note: My forecast looks at the end of period values always. The bookings forecast is quarterly, which we change to be a monthly (level) bookings forecast. To arrive at the value of the bookings forecast at the end of the first period, one period's worth (1/3) of a qarter, has already been booked and need to be eliminated from the dataframe.
    #
    # To adjust for this, the bookings dataframe will have a P0 that contains the first month's bookings and will be removed.
    #
    # This created an error the first time I tested the program that overstated the bookings, billings and deferred.


    new_columns = []
    for i in range(12 * 3):
        if len(str(i)) == 1:
            new_column = "P0" + str(i)
        else:
            new_column = "P" + str(i)
        new_columns.append(new_column)

    qtrly_list = [col for col in df_test_gb.columns if "Q" in col]
    qtrly_list


    period_index = 0
    for index, qtr in enumerate(qtrly_list):

        df_test_gb[new_columns[period_index]] = df_test_gb[qtr] / 3
        period_index += 1
        df_test_gb[new_columns[period_index]] = df_test_gb[qtr] / 3
        period_index += 1
        df_test_gb[new_columns[period_index]] = df_test_gb[qtr] / 3
        period_index += 1


    df_test_gb


    # ## I Don't need the everything in this. I can now remove some of the details
    #
    # First check that my periods match the quarterly deferred numbers/


    df_qtrly_only = df_test_gb.copy()
    df_period_only = df_test_gb.copy()

    df_period_only = df_period_only.loc[:, df_period_only.columns.str.contains("P")]
    df_qtrly_only = df_qtrly_only.loc[:, ~df_qtrly_only.columns.str.contains("P")]

    df_period_only["total"] = df_period_only.sum(axis=1)


    df_period_only


    df_qtrly_only


    # ##### OK My periods work fine. Now I can move on to saving this and finishing the defered waterfall

    df_test_gb.columns

    df_waterfall = df_test_gb.loc[:, df_test_gb.columns.str.contains("P")]


    df_waterfall


    # ##### Now dropping P0 from the waterfall


    df_waterfall = df_waterfall.drop("P00", axis=1)


    return df_waterfall




#### End of function

# #### Saving the waterfall as Q2_waterfall


pickle.dump(df_waterfall, open("../data/processed/Q2_waterfall.p", "wb"))


# ## Building the Deferred Revenue Waterfall from the forecast dataframe (df_fcst) and the waterfall dataframe (df_waterfall)

df_waterfall.head(10)


# # NOTE: I am ignoring Deferred Type B (Service) billings - for now

# ### Deferred Revenue Assumptions
#
# ##### Monthly Deferred Billings
# These occur in the middle of the month. Half the billings go directly to revenue, the remainder amortize out of deferred the next month
#
# ##### Three Month Deferred Billings
# These are assumed to occur at the end of the period.
#
#
# ##### Annual Billings
# - 1/12 of the current annual billings + 11 of the last annual billings + 1/12 of the year prior billings
#


df_wf = build_deferred_waterfall(df_fcst)


df_wf = df_wf.reset_index(drop=True)


# ### The billings file is at the Enterprise BU level and the bookings forecast is at the BU level


df_wf["BU"] = df_wf["BU"].str.replace("Creative", "Digital Media")
df_wf["BU"] = df_wf["BU"].str.replace("Document Cloud", "Digital Media")
df_wf["BU"] = df_wf["BU"].str.replace("DX Other", "Digital Experience")
df_wf["BU"] = df_wf["BU"].str.replace("Experience Cloud", "Digital Experience")
df_wf["BU"] = df_wf["BU"].str.replace("Print & Publishing", "Publishing")


df_wf.head(40)


# ##### Testing one slice of the dataframe


this_curr = "EUR"
this_BU = "Digital Experience"
this_slice = df_wf[(df_wf["BU"] == this_BU) & (df_wf["curr"] == this_curr)]
this_slice


# ### So I have the deferred waterfall by DC, period, BU.
# # I need to groupby BU and Period


df_wf_gb = df_wf.groupby(["BU", "period"]).sum()


df_wf_gb.head(40)


# ### Now we need to group the BU to match the waterfall BUs


df_wf_gb.index


df_waterfall


df_wf_gb.reset_index(inplace=True)


df_wf_gb.head(5)


# ### Change the BU to match the waterfall BUs
#  - Creative to Digital Media
#  - Document Cloud to Digital Media
#  - Print & Publishing to Publishing
#  - DX Other to Digital Experience
#  - Experience Cloud to Digital Experience


new_slice = df_wf_gb[df_wf_gb["BU"] == "Digital Experience"]
new_slice


# ## Altering the initial waterfall fields


df_waterfall

df_waterfall.drop("Grand inclusive of Magento/Marketo", inplace=True)


# ## Take the As Performed / Upon Acceptance column and place this into the df_wf_gb dataframe.
# ## We will assume that this does not change over time


df_as_performed = df_waterfall["As Performed / Upon Acceptance"].copy()
df_as_performed


df_waterfall = df_waterfall.drop("As Performed / Upon Acceptance", axis=1)


df_waterfall


# ## Changing the periods in the df_wf_gb to match the df_watefall first


old_cols = df_wf_gb.columns
old_cols = old_cols[3:]


new_columns = []
for i in range(12 * 3):
    if len(str(i + 1)) == 1:
        new_column = "P0" + str(i + 1)
    else:
        new_column = "P" + str(i + 1)
    new_columns.append(new_column)


rename_dict = dict(zip(old_cols, new_columns))
df_wf_gb = df_wf_gb.rename(columns=rename_dict)
df_wf_gb


list_periods = df_wf_gb.period.unique()
list_periods


df_waterfall

list_BU = df_wf_gb.BU.unique()
list_BU


df_waterfall["period"] = "2020-07"
df_waterfall

df_waterfall = df_waterfall.reset_index()
df_waterfall


df_waterfall.rename(columns={"External Reporting BU": "BU"}, inplace=True)
df_waterfall


# ## ADDING ADDITONAL PERIODS HERE TO MERGE WITH df_wf


df_waterfall["P28"] = 0
df_waterfall["P29"] = 0
df_waterfall["P30"] = 0
df_waterfall["P31"] = 0
df_waterfall["P32"] = 0
df_waterfall["P33"] = 0
df_waterfall["P34"] = 0
df_waterfall["P35"] = 0
df_waterfall["P36"] = 0


df_waterfall.head(10)


# ##### Planning on making a new dataframe with BU and period and merging them. Will create NAs everywhere else, but we will fillna


list_periods = list_periods[1:]
list_periods


list_BU


to_df_BU = []
to_df_period = []
for item in list_BU:

    for per in list_periods:
        to_df_BU.append(item)
        to_df_period.append(per)


len(to_df_BU)


# ### Creating new dataframe to be merged


df_to_merge = pd.DataFrame({"BU": to_df_BU, "period": to_df_period})


df_waterfall = df_waterfall.merge(df_to_merge, on=["BU", "period"], how="outer")


df_waterfall.fillna(0, inplace=True)


df_waterfall.sort_values(by=["BU", "period"], inplace=True)


df_waterfall


# ## Now move the waterfall forward by BU
df_waterfall = bring_initial_wf_forward(df_waterfall)

df_waterfall


df_wf_gb.columns


df_waterfall.columns


df_waterfall = df_waterfall.set_index(["BU", "period"])


df_wf_gb = df_wf_gb.set_index(["BU", "period"])


df_all = df_waterfall.add(df_wf_gb, fill_value=0)


df_waterfall = df_waterfall.sort_index()
df_all = df_all.sort_index()
df_wf_gb = df_wf_gb.sort_index()


df_all.head(40)


df_wf_gb.head(10)


df_waterfall.head(10)


# #### Sending this data over to excel as a check


with pd.ExcelWriter("output.xlsx") as writer:
    df_waterfall.to_excel(writer, sheet_name="initial_waterfall")
    df_wf_gb.to_excel(writer, sheet_name="billings_impact")
    df_all.to_excel(writer, sheet_name="combined")
    df_wf.to_excel(writer, sheet_name="early_wf")


df_waterfall.columns


# ## Add the as performed back into the waterfall forecast


df_all["Total"] = df_all[df_all.columns[:-1]].sum(axis=1)


df_waterfall["Total"] = df_waterfall[df_waterfall.columns[:]].sum(axis=1)


df_wf_gb.columns

df_wf_gb["Total"] = df_wf_gb[df_wf_gb.columns[1:]].sum(axis=1)


# # Need to add these back to our model dictionary
# ## At this point I dont recall what it was called and will need to change it


saved_dict["waterfall"] = df_all
saved_dict["bill_waterfall"] = df_wf_gb
saved_dict["initial_waterfall"] = df_waterfall


pickle.dump(saved_dict, open("../data/processed/final_forecast_2.p", "wb"))


# ### Testing parts of the bookings/waterfall


list_Q3 = ["2020-07", "2020-08", "2020-09"]
this_BU = "Creative"
test_Q3 = df[(df["BU"] == this_BU) & (df["period"].isin(list_Q3))]
test_Q3["book_1Y_US"].sum()


list_Q4 = ["2020-10", "2020-11", "2020-12"]
this_BU = "Creative"
test_Q4 = df[(df["BU"] == this_BU) & (df["period"].isin(list_Q4))]


test_Q4["book_1Y_US"].sum()


saved_dict["waterfall"] = df_all
saved_dict["bill_waterfall"] = df_wf_gb
saved_dict["initial_waterfall"] = df_waterfall


# ##### Merging the df_fcst with the df_bililngs for easier charting?


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
    "initial_waterfall": df_waterfall,
    "bill_waterfall": df_wf_gb,
    "waterfall": df_all,
}
pickle.dump(input_df_dict, open("../data/processed/final_forecast.p", "wb"))


# %%
