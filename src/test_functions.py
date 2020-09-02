
import pandas as pd
import numpy as np

df = pd.read_excel("data/Data_2020_P06/Q2'20 Rev Acctg Mgmt Workbook (06-04-20).xlsx",
                   sheet_name='Deferred Revenue Forecast', skiprows=5)

df.head(50)












"""
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
"""


""" Need to build a simple test of the booking periods to make sure they are correct """
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


'''
Testing the billings forecast to make this thing a function

'''
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

def setting_up_fcst_df(df_billings, df_cal, config_dict):
    # This function sets up a dataframe to be used for the df_fcst
    # We need three variables -
    # 1. BUs
    # 2. currencies
    # 3. periods
    #
    # creating the list of historical bill periods
    #
    # INPUTS
    #   df_billings
    #   config_dict
    # OUTPUT
    #   df_fcst (a dataframe of zeros needed to
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

    # Creating a list of the columns that we need to use in the df_billings dataframe (They contain document currency billings)

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

    return df_fcst


['output_dir']['intermediate'] + "int_output_1.p"