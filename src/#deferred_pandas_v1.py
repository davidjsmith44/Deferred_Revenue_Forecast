# deferred_pandas_v1.py

"""
Describe data sources here




Describe process in steps here




output explained here



Later these will be made into easier to use functions and imported into a simpler program






"""

# list of functions
def remove_infreq_curr(df, count_threshold=10):
    """
    There are many currencies that only appear a few times in the billings dataset and we are removing them due to their infrequenct billings.
    For example, there is one MXP transaction in the four year history of billings.

    NOTE: As of April 2020, the TRY currency does not have forward rates or exchange rates in our dataset,
    so we are temporarilly removing this currency as well.

    INPUT:
        df:                 the original billings database right after it is loaded into a dataframe

        count_threshold:    the cut off number of dates to remove a currency (default = 10)

    OUTPUT:

        df:          the billings dataframe with the less frequent  currencies removed

        model_dict:  a dictionary being used as the 'key' storing all relevant model information
                        'curr_removed' - is a list of the currencies removed by this function

    """

    vc = df["curr"].value_counts()
    keep_these = vc.values > count_threshold
    keep_curr = vc[keep_these]
    a = keep_curr.index
    df = df[df["curr"].isin(a)]

    # keeping track of the currencies that have been removed in the model_dict
    remove_these = vc[vc.values <= 10].index
    model_dict = {"curr_removed": list(vc[remove_these].index)}
    delete_curr = list(remove_these)

    # removing the 'TRY' currency due to lack of FX data
    if "TRY" not in model_dict:
        model_dict["curr_removed"].append("TRY")
        delete_curr.append("TRY")
        a = a.drop("TRY")

    # standard reporting metrics printed to screen
    print("Model dictionary", model_dict)
    print("Deleted Currencies", delete_curr)

    print("---Removing infrequent currencies from billings history---")
    print("Total number of currencies in the base billings file: ", len(vc))
    if len(model_dict["curr_removed"]) == 0:
        print("No currencies were removed, all contained 10 or more billings")
        print("Currencies in the base billings file")
        for item in a:
            print(a[item], end=" ")
    else:
        print("\n Currencies were removed: ", len(model_dict["curr_removed"]))

        for item in remove_these:
            print(item, ", ", end="")

        print("\n\n{} Remaining currencies: ".format(len(a)))
        for item in a:
            print(item, ", ", end="")

    return df, model_dict


def clean_df_columns(df):

    # clean up NaNs before adding
    df = df.fillna(value=0)

    # DC columns first
    # Monthly
    df["deferred_1M_DC"] = df["deferred_1M_a_DC"] + df["deferred_1M_d_DC"]
    df.drop(labels=["deferred_1M_a_DC", "deferred_1M_d_DC"], axis=1, inplace=True)

    # Annual
    df["deferred_1Y_DC"] = df["deferred_1Y_a_DC"] + df["deferred_1Y_d_DC"]
    df.drop(labels=["deferred_1Y_a_DC", "deferred_1Y_d_DC"], axis=1, inplace=True)

    # Two-Year
    df["deferred_2Y_DC"] = df["deferred_2Y_a_DC"] + df["deferred_2Y_d_DC"]
    df.drop(labels=["deferred_2Y_a_DC", "deferred_2Y_d_DC"], axis=1, inplace=True)

    # Three-Year
    df["deferred_3Y_DC"] = df["deferred_3Y_a_DC"] + df["deferred_3Y_d_DC"]
    df.drop(labels=["deferred_3Y_a_DC", "deferred_3Y_d_DC"], axis=1, inplace=True)

    # renaming 3M and 6M
    df.rename(
        index=str,
        columns={
            "deferred_3M_d_DC": "deferred_3M_DC",
            "deferred_6M_d_DC": "deferred_6M_DC",
        },
        inplace=True,
    )

    # US columns
    # Monthly
    df["deferred_1M_US"] = df["deferred_1M_a_US"] + df["deferred_1M_d_US"]
    df.drop(labels=["deferred_1M_a_US", "deferred_1M_d_US"], axis=1, inplace=True)

    # Annual
    df["deferred_1Y_US"] = df["deferred_1Y_a_US"] + df["deferred_1Y_d_US"]
    df.drop(labels=["deferred_1Y_a_US", "deferred_1Y_d_US"], axis=1, inplace=True)

    # Two-Year
    df["deferred_2Y_US"] = df["deferred_2Y_a_US"] + df["deferred_2Y_d_US"]
    df.drop(labels=["deferred_2Y_a_US", "deferred_2Y_d_US"], axis=1, inplace=True)

    # Three-Year
    df["deferred_3Y_US"] = df["deferred_3Y_a_US"] + df["deferred_3Y_d_US"]
    df.drop(labels=["deferred_3Y_a_US", "deferred_3Y_d_US"], axis=1, inplace=True)

    # renaming 3M and 6M
    df.rename(
        index=str,
        columns={
            "deferred_3M_d_US": "deferred_3M_US",
            "deferred_6M_d_US": "deferred_6M_US",
        },
        inplace=True,
    )

    # cleaning up the longer column names
    df.rename(
        index=str,
        columns={"curr": "curr", "BU": "BU", "period": "period"},
        inplace=True,
    )

    return df


def merge_new_dataframe(old_df, new_df, new_column):
    df_merged = pd.merge(
        old_df,
        new_df,
        how="outer",
        left_on=["curr", "BU", "period"],
        right_on=["curr", "BU", "period"],
    )
    df_merged.rename(
        index=str,
        columns={"DC_amount": new_column + "_DC", "US_amount": new_column + "_US"},
        inplace=True,
    )

    # need to drop the product configtype id for merges where the new_df is of type A
    config_str = "config"
    rule_str = "rebill_rule"
    if config_str in df_merged.columns:
        df_merged.drop(columns=["config"], inplace=True)

    if rule_str in df_merged.columns:
        df_merged.drop(columns=["rebill_rule"], inplace=True)

    return df_merged


def sum_USD_amt(list_df, list_columns):
    total_US = []
    for df in list_df:
        total_US.append(df["US_amount"].sum())
    total_df = pd.DataFrame(index=list_columns, columns=["US_amounts"], data=total_US)
    return total_df


def merge_all_dataframes(list_df, list_columns):
    for i, df in enumerate(list_df):
        print("This is i:", i)
        # print("This is the df: ", df.head())
        print("referencing the column: ", list_columns[i])

        if i == 0:
            df_merged = list_df[0].copy()
            df_merged.rename(
                index=str,
                columns={
                    "DC_amount": list_columns[i] + "_DC",
                    "US_amount": list_columns[i] + "_US",
                },
                inplace=True,
            )
        else:
            df_merged = merge_new_dataframe(df_merged, df, list_columns[i])

    return df_merged


# import statements
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from math import ceil

# load up the base billings file
df = pd.read_excel("../data/all_billings_inputs.xlsx", sheet_name="base_billings")

# Rename the columns
df.rename(
    index=str,
    columns={
        "Document Currency": "curr",
        "Enterprise BU Desc": "BU",
        "Invoice Fiscal Year Period Desc": "period",
        "Product Config Type": "config",
        "Rev Rec Category": "rev_req_type",
        "Rule For Bill Date": "rebill_rule",
        "Completed Sales ( DC )": "DC_amount",
        "Completed Sales": "US_amount",
    },
    inplace=True,
)


df, model_dict = remove_infreq_curr(df)

# remove any values that are zero
print("This is the length of the dataframe before removing zeros: ", len(df))
df = df[df["DC_amount"] != 0]
print("This is the length of the dataframe AFTER removing zeros: ", len(df))

# Remove any billings that are 'NON-REV' sales type
print("Length of the dataframe before removing non-revenue billings: ", len(df))
df = df[df["Sales Type"] != "NON-REV"]
print("Length of the dataframe after removing non-revenue billings:  ", len(df))

# starting split - apply - combine in pandas
# split into sales type dataframes
rec = df[df["Sales Type"] == "RECOGNIZED"].copy()
svc = df[df["Sales Type"] == "PRO-SVC-INV"].copy()
dfr = df[df["Sales Type"] == "DEFERRED"].copy()

# RECOGNIZED REVENUE
# NOTE: The subscription term is the only numeric field that stays after the groupby completed.
gb_rec = rec.groupby(["curr", "BU", "period"], as_index=False).sum()
gb_rec.drop(labels="Subscription Term", axis=1, inplace=True)

# SERVICE BASED BILLINGS
gb_svc = svc.groupby(["curr", "BU", "period"], as_index=False).sum()
gb_svc.drop(labels="Subscription Term", axis=1, inplace=True)

# DEFERRED BILLINGS
# Type B: filter out the type B first then do a group_by
dfr_b = dfr[dfr["rev_req_type"] == "B"].copy()
gb_b = dfr_b.groupby(["curr", "BU", "period"], as_index=False).sum()
gb_b.drop(labels="Subscription Term", axis=1, inplace=True)

print("length of deferred billings : ", len(dfr))
print("length of the type B billings: ", len(dfr_b))

# Type A:
"""
Type A billings have a billing plan that specifies the time between billings. {'3Y', '2Y', '1Y', 'MTHLY'}
"""
dfr_a = dfr[dfr["rev_req_type"] == "A"].copy()
gb_a = dfr_a.groupby(["curr", "BU", "period", "config"], as_index=False).sum()
gb_a.drop(labels="Subscription Term", axis=1, inplace=True)

config_list = ["1Y", "2Y", "3Y", "MTHLY"]
df_temp1 = gb_a[gb_a["config"].isin(config_list)]

gb_a_1Y = df_temp1[df_temp1["config"] == "1Y"].copy()
gb_a_2Y = df_temp1[df_temp1["config"] == "2Y"].copy()
gb_a_3Y = df_temp1[df_temp1["config"] == "3Y"].copy()
gb_a_1M = df_temp1[df_temp1["config"] == "MTHLY"].copy()

print("this is the lenght of type A 1M billings: ", len(gb_a_1M))
print("this is the lenght of type A 1Y billings: ", len(gb_a_1Y))
print("this is the lenght of type A 2Y billings: ", len(gb_a_2Y))
print("this is the lenght of type A 3Y billings: ", len(gb_a_3Y))

# Type D:
"""
Type D billings have a rule for bill data that determines when the contract rebills
    Monthly:        {Y1, Y2, Y3, Y5}
    Quarterly:      YQ
    Every 4 months: YT     NOTE: These are treated as quarterly because they are small in number and amount
    Semi-annual:    YH
    Annual:         {YA, YC}
    2 years:        Y4
    3 years:        Y7
"""
dfr_d = dfr[dfr["rev_req_type"] == "D"].copy()
gb_d = dfr_d.groupby(["curr", "BU", "period", "rebill_rule"], as_index=False).sum()
gb_d.drop(labels="Subscription Term", axis=1, inplace=True)


gb_d_mthly = gb_d[gb_d["rebill_rule"].isin(["Y1", "Y2", "Y3", "YM"])].copy()
gb_d_mthly.drop(labels="rebill_rule", axis=1, inplace=True)
gb_d_mthly = gb_d_mthly.groupby(["curr", "BU", "period"]).sum()
gb_d_mthly.reset_index(inplace=True)

gb_d_qtrly = gb_d[gb_d["rebill_rule"].isin(["YQ", "YY", "YT"])].copy()
gb_d_qtrly.drop(labels="rebill_rule", axis=1, inplace=True)
gb_d_qtrly = gb_d_qtrly.groupby(["curr", "BU", "period"]).sum()
gb_d_qtrly.reset_index(inplace=True)

gb_d_semi_ann = gb_d[gb_d["rebill_rule"] == "YH"]

gb_d_annual = gb_d[gb_d["rebill_rule"].isin(["YA", "YC", "YX"])].copy()
gb_d_annual.drop(labels="rebill_rule", axis=1, inplace=True)
gb_d_annual = gb_d_annual.groupby(["curr", "BU", "period"]).sum()
gb_d_annual.reset_index(inplace=True)


gb_d_two_yrs = gb_d[gb_d["rebill_rule"] == "Y4"]
gb_d_three_yrs = gb_d[gb_d["rebill_rule"] == "Y7"]

print("Length of monthly", len(gb_d_mthly))
print("Length of quarterly", len(gb_d_qtrly))
print("Length of semi ann", len(gb_d_semi_ann))
print("Length of annual", len(gb_d_annual))
print("Length of two years", len(gb_d_two_yrs))
print("Length of three years", len(gb_d_three_yrs))
"""
NOW WE NEED TO BUILD A DATAFRAME THAT INTEGRATES THIS DATA

- We will have the following descriptive fields
   - Invoicing Fiscal Year-Period
   - Document Currency
   - Enterprise BU

- We will have the following fields based on rebilling rule
   - Recognized
   - Service
   - Monthly
   - Quarterly
   - Annual
   - Two Years
   - Three Years
"""

# We need to do it this way when we get to a .py file!
list_df = [
    gb_rec,
    gb_svc,
    gb_b,
    gb_a_1M,
    gb_a_1Y,
    gb_a_2Y,
    gb_a_3Y,
    gb_d_mthly,
    gb_d_qtrly,
    gb_d_semi_ann,
    gb_d_annual,
    gb_d_two_yrs,
    gb_d_three_yrs,
]

list_columns = [
    "recognized",
    "service",
    "deferred_B",
    "deferred_1M_a",
    "deferred_1Y_a",
    "deferred_2Y_a",
    "deferred_3Y_a",
    "deferred_1M_d",
    "deferred_3M_d",
    "deferred_6M_d",
    "deferred_1Y_d",
    "deferred_2Y_d",
    "deferred_3Y_d",
]


df = merge_all_dataframes(list_df, list_columns)

df = clean_df_columns(df)

# ZCC BILLINGS
# loading Adobe financial calendar and calculating period weeks
df_cal = pd.read_excel("../data/old/ADOBE_FINANCIAL_CALENDAR.xlsx", "ADBE_cal")
df_cal["Period_Weeks"] = (df_cal["Per_End"] - df_cal["Per_Start"]) / np.timedelta64(
    1, "W"
)
df_cal["Period_Weeks"] = df_cal["Period_Weeks"].astype(int)
df_cal["Period_Weeks"] = df_cal["Period_Weeks"] + 1
df_cal["p2digit"] = df_cal["Period"].astype(str)
df_cal["p2digit"] = df_cal["p2digit"].str.zfill(2)
df_cal["period_match"] = (
    df_cal["Year"].astype(str) + "-" + df_cal["p2digit"].astype(str)
)

df_cal.drip(["p2digit"], axis=1, inplace=True)

df_A = pd.read_excel("../data/all_billings_inputs.xlsx", sheet_name="type_A_no_config")

df_A.rename(
    index=str,
    columns={
        "Document Currency": "curr",
        "Enterprise BU Desc": "BU",
        "Invoice Fiscal Year Period Desc": "period",
        "Rev Rec Contract End Date Hdr": "end_date_1",
        "Rev Rec Contract End Date Item": "end_date_2",
        "Rev Rec Contract Start Date Hdr": "start_date_1",
        "Rev Rec Contract Start Date Item": "start_date_2",
        "Completed Sales ( DC )": "DC_amount",
        "Completed Sales": "US_amount",
    },
    inplace=True,
)

df_A["start_date_str"] = df_A[["start_date_1", "start_date_2"]].max(axis=1).astype(str)
df_A["end_date_str"] = df_A[["end_date_1", "end_date_2"]].max(axis=1).astype(str)

df_A["start_date"] = pd.to_datetime(df_A["start_date_str"])
df_A["end_date"] = pd.to_datetime(df_A["end_date_str"])

df_A.drop(
    labels=[
        "end_date_1",
        "end_date_2",
        "start_date_1",
        "start_date_2",
        "start_date_str",
        "end_date_str",
    ],
    axis=1,
    inplace=True,
)

df_A["month_interval"] = df_A["end_date"] - df_A["start_date"]
df_A["months"] = (df_A["month_interval"] / np.timedelta64(1, "M")).round(0)

list_rebills = [1, 3, 6, 12, 24, 36]
temp_rebill = np.zeros_like(df_A["months"])
for i in range(len(df_A)):
    temp_rebill[i] = min(list_rebills, key=lambda x: abs(x - df_A["months"][i]))
df_A["rebill_months"] = temp_rebill

df_A.drop(
    columns=["start_date", "end_date", "month_interval", "months"], axis=1, inplace=True
)

# pivoting on the rebill_months, filling in any blanks and flattening back to a dataframe
temp_DC = df_A.pivot_table("DC_amount", ["curr", "BU", "period"], "rebill_months")
temp_US = df_A.pivot_table("US_amount", ["curr", "BU", "period"], "rebill_months")
temp_DC = temp_DC.fillna(0)
temp_US = temp_DC.fillna(0)
temp_flat_DC = pd.DataFrame(temp_DC.to_records())
temp_flat_US = pd.DataFrame(temp_US.to_records())

temp_flat_DC.rename(
    index=str,
    columns={
        "1.0": "deferred_1M_DC",
        "3.0": "deferred_3M_DC",
        "6.0": "deferred_6M_DC",
        "12.0": "deferred_1Y_DC",
        "24.0": "deferred_2Y_DC",
        "36.0": "deferred_3Y_DC",
    },
    inplace=True,
)

temp_flat_US.rename(
    index=str,
    columns={
        "1.0": "deferred_1M_US",
        "3.0": "deferred_3M_US",
        "6.0": "deferred_6M_US",
        "12.0": "deferred_1Y_US",
        "24.0": "deferred_2Y_US",
        "36.0": "deferred_3Y_US",
    },
    inplace=True,
)

# now merge the dataframes
df_with_A = pd.merge(
    df,
    temp_flat_DC,
    how="outer",
    left_on=["curr", "BU", "period"],
    right_on=["curr", "BU", "period"],
    indicator=True,
    validate="one_to_one",
)

# filling in NaNs for the numeric columns
df_with_A = df_with_A.fillna(
    pd.Series(0, index=df_with_A.select_dtypes(exclude="category").columns)
)

# now merging the US billings
df_with_all = pd.merge(
    df_with_A,
    temp_flat_US,
    how="outer",
    left_on=["curr", "BU", "period"],
    right_on=["curr", "BU", "period"],
)
df_with_all = df_with_all.fillna(
    pd.Series(0, index=df_with_all.select_dtypes(exclude="category").columns)
)

df_with_all["deferred_1M_DC"] = (
    df_with_all["deferred_1M_DC_x"] + df_with_all["deferred_1M_DC_y"]
)
df_with_all["deferred_3M_DC"] = (
    df_with_all["deferred_3M_DC_x"] + df_with_all["deferred_3M_DC_y"]
)
df_with_all["deferred_6M_DC"] = (
    df_with_all["deferred_6M_DC_x"] + df_with_all["deferred_6M_DC_y"]
)
df_with_all["deferred_1Y_DC"] = (
    df_with_all["deferred_1Y_DC_x"] + df_with_all["deferred_1Y_DC_y"]
)
df_with_all["deferred_2Y_DC"] = (
    df_with_all["deferred_2Y_DC_x"] + df_with_all["deferred_2Y_DC_y"]
)
df_with_all["deferred_3Y_DC"] = (
    df_with_all["deferred_3Y_DC_x"] + df_with_all["deferred_3Y_DC_y"]
)

df_with_all["deferred_1M_US"] = (
    df_with_all["deferred_1M_US_x"] + df_with_all["deferred_1M_US_y"]
)
df_with_all["deferred_3M_US"] = (
    df_with_all["deferred_3M_US_x"] + df_with_all["deferred_3M_US_y"]
)
df_with_all["deferred_6M_US"] = (
    df_with_all["deferred_6M_US_x"] + df_with_all["deferred_6M_US_y"]
)
df_with_all["deferred_1Y_US"] = (
    df_with_all["deferred_1Y_US_x"] + df_with_all["deferred_1Y_US_y"]
)
df_with_all["deferred_2Y_US"] = (
    df_with_all["deferred_2Y_US_x"] + df_with_all["deferred_2Y_US_y"]
)
df_with_all["deferred_3Y_US"] = (
    df_with_all["deferred_3Y_US_x"] + df_with_all["deferred_3Y_US_y"]
)

df_with_all.drop(
    labels=[
        "deferred_1M_DC_x",
        "deferred_1M_DC_y",
        "deferred_3M_DC_x",
        "deferred_3M_DC_y",
        "deferred_6M_DC_x",
        "deferred_6M_DC_y",
        "deferred_1Y_DC_x",
        "deferred_1Y_DC_y",
        "deferred_2Y_DC_x",
        "deferred_2Y_DC_y",
        "deferred_3Y_DC_x",
        "deferred_3Y_DC_y",
        "deferred_1M_US_x",
        "deferred_1M_US_y",
        "deferred_3M_US_x",
        "deferred_3M_US_y",
        "deferred_6M_US_x",
        "deferred_6M_US_y",
        "deferred_1Y_US_x",
        "deferred_1Y_US_y",
        "deferred_2Y_US_x",
        "deferred_2Y_US_y",
        "deferred_3Y_US_x",
        "deferred_3Y_US_y",
    ],
    axis=1,
    inplace=True,
)

# print_check
print("sum of temp flat DC 1M", temp_flat_DC["deferred_1M_DC"].sum())
print("sum of base_df before DC 1M", df["deferred_1M_DC"].sum())
print("sum of final DC 1M", df_with_all["deferred_1M_DC"].sum())

a = temp_flat_DC["deferred_1M_DC"].sum()
b = df["deferred_1M_DC"].sum()
c = df_with_all["deferred_1M_DC"].sum()
print(c)
print(a + b)

# now creating a dictionary that contains all of the input data we need
df = df_with_all.copy()

# currency map
df_curr_map = pd.read_excel("../data/currency_map.xlsx", sheet_name="curr_map")
df_curr_map["Country"] = df_curr_map["Country"].str.replace("\(MA\)", "", case=False)

# FX spot rates, forward rates and vols
df_FX_rates = pd.read_excel("../data/FX_data.xlsx", sheet_name="to_matlab")
df_FX_rates["VOL_3M"] = df_FX_rates["VOL_3M"] / 100
df_FX_rates["VOL_6M"] = df_FX_rates["VOL_6M"] / 100
df_FX_rates["VOL_9M"] = df_FX_rates["VOL_9M"] / 100
df_FX_rates["VOL_1Y"] = df_FX_rates["VOL_1Y"] / 100

# FX forward rates used in the plan
df_FX_fwds = pd.read_excel(
    "../data/FX_forward_rates.xlsx",
    sheet_name="forward_data",
    skiprows=1,
    usecols="C,G",
)
df_FX_fwds.rename(
    index=str, columns={"Unnamed: 2": "curr", "FWD REF": "forward"}, inplace=True
)

# loading the bookings
df_bookings = pd.read_excel("../data/2020_bookings_fcst_Q1.xlsx", sheet_name="bookings")

# cleaning up the bookings
"""
The bookings file contains odd test in parenthesis that needs to be removed
"""
df_bookings["EBU"] = df_bookings["EBU"].str.replace(" \(EB\)", "", case=False)
df_bookings["Internal Segment"] = df_bookings["Internal Segment"].str.replace(
    "\(IS\)", ""
)

df_bookings["PMBU"] = df_bookings["PMBU"].str.replace("\(PMBU\)", "")
df_bookings["GEO"] = df_bookings["GEO"].str.replace("\(G\)", "")
df_bookings["Market Area"] = df_bookings["Market Area"].str.replace("\(MA\)", "")


df_bookings.drop(
    columns=["Hedge", "Mark Segment", "Type", "Scenario", "FX"], inplace=True
)

df_bookings.rename(
    index=str,
    columns={
        "EBU": "BU",
        "Internal Segment": "segment",
        "PMBU": "product",
        "GEO": "geo",
        "Market Area": "country",
        "Bookings Type": "booking_type",
        "value": "US_amount",
    },
    inplace=True,
)

list_book_ctry = df_bookings["country"].unique()
list_curr_map = df_curr_map["Country"].unique()
set_in_both = list(set(list_book_ctry) & set(list_curr_map))
not_in_map = set(list_book_ctry).difference(set(list_curr_map))
if len(not_in_map) > 0:
    print(
        "We have a mismatch between the bookings countries and the currency_map countries"
    )
    print("This is the set that is not in the currency map: ", not_in_map)

# merge the currency map currencies into the df_bookings dataframe
df_bookings = pd.merge(
    df_bookings, df_curr_map, how="left", left_on="country", right_on="Country"
)


# taking the period weeks from the calendar and putting this on the df (billings) and df_fcst
df_cal_2_merge = df_cal.copy()
df_cal_2_merge.drop(
    [
        "Year",
        "Quarter",
        "Period",
        "Qtr_Ticker",
        "Qtr_Start",
        "Qtr_End",
        "Per_Start",
        "Per_Ticker",
        "Per_End",
    ],
    axis=1,
    inplace=True,
)
df = df.merge(df_cal_2_merge, how="left", left_on="period", right_on="period_match")
df.drop(["period_match", "_merge"], axis=1, inplace=True)


input_df_dict = {
    "billings": df,
    "ADBE_cal": df_cal,
    "bookings": df_bookings,
    "FX_forwards": df_FX_fwds,
    "FX_rates": df_FX_rates,
}

# pickle.dump(input_df_dict, open('../data/processed/all_inputs.p', 'wb'))

df_billings = df.copy()
