"""
Trying to recreate the deferred revenue waterfall using Pandas exclusively


"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
move to below later
df = pd.read_excel("../data/old/base_billings.xlsx", sheet_name="bill_DC")
"""


def load_base_billings(filename, sheetname):
    """
    This function reads the excel file that contains the base billings in document currency
    and returns a dataframe that is formatted like -------------

    INPUTS:
        filename: a string containing the path to the base billings excel file
        sheetname: the sheetname that contains the data for the document currency
    OUTPUT:
        df  a dataframe containing the following



    STEPS:
        1. Remove currencies that have less than 10 transactions
        2. Remove values that are zero
        3. Clear out any Non-Revenue Billings
        4. Split Apply Combine on the dataframes

    """
    df = pd.read_excel("../data/old/base_billings.xlsx", sheet_name="bill_DC")

    df.rename(
        index=str,
        columns={
            "Document Currency": "curr",
            "Enterprise Bu": "BU",
            "Invoicing Fiscal Year-Period Desc": "period",
            "Product Configtype ID": "config",
            "Revenue Recognition Category New": "rev_req_type",
            "Rule For Bill Date": "rebill_rule",
            "Completed Sales Doc Currency": "amount",
        },
        inplace=True,
    )

    # Step 1  Remove any currencies that have less than 10 transaction
    vc = df["Document Currency"].value_counts()
    keep_these = vc.values > 10
    keep_curr = vc[keep_these]
    a = keep_curr.index
    df = df[df["Document Currency"].isin(a)]
    remove_these = vc.values <= 10
    model_dict = {"curr_removed": list(vc[remove_these].index)}

    # Need to create some sort of a report with respect to the currencies removed and kept
    print("---Removing infrequent currencies from billings history---")
    print("Total number of currencies in the base billings file: ", len(vc))
    if sum(remove_these) == 0:
        print("No currencies were removed, all contained 10 or more billings")
        print("Currencies in the base billings file")
        for item in a:
            print(a[item], end=" ")
    else:
        print("\n{} Currencies were removed: ".format(sum(remove_these)))

        for item in vc[remove_these].index:
            print(item, ", ", end="")

        print("\n\n{} Remaining currencies: ".format(len(a)))
        for item in a:
            print(item, ", ", end="")

    # Step 2: Remove any values in the dataframe that are zero
    print("Removing Zeros from the billings file")
    print("This is the length of the dataframe before removing zeros: ", len(df))
    df = df[df["amount"] != 0]
    print("This is the length of the dataframe after removing zeros: ", len(df))

    # Step 3: Clear out any Non-Revenue billings from the file
    df["Sales Type"].value_counts()
    print("Length of the dataframe before removing non-revenue billings: ", len(df))
    df = df[df["Sales Type"] != "NON-REV"]
    print("Length of the dataframe after removing non-revenue billings:  ", len(df))

    # Step 4: Split Apply Combine on the dataframes
    # First split the data into three dataframes
    rec = df[df["Sales Type"] == "RECOGNIZED"]
    svc = df[df["Sales Type"] == "PRO-SVC-INV"]
    dfr = df[df["Sales Type"] == "DEFERRED"]

    # RECOGNIZED BILLINGS
    gb_rec = rec.groupby(["curr", "BU", "period"], as_index=False,).sum()
    gb_rec.drop(labels="Subscription Term", axis=1, inplace=True)

    # SERVICE BASED BILLINGS
    gb_svc = svc.groupby(["curr", "BU", "Period"], as_index=False,).sum()
    gb_svc.drop(labels="Subscription Term", axis=1, inplace=True)

    # DEFERRED BILLINGS
    # Service Based Deferred Billings: Type B
    dfr_b = dfr[dfr["rev_req_type"] == "B"]
    gb_b = dfr_b.groupby(["curr", "BU", "Period"], as_index=False,).sum()
    gb_b.drop(labels="Subscription Term", axis=1, inplace=True)

    print("length of deferred billings : ", len(dfr))
    print("length of the type B billings: ", len(dfr_b))

    # Type A Deferred Billings
    dfr_a = dfr[dfr["rev_req_type"] == "A"]
    gb_a = dfr_a.groupby(["Curr", "BU", "Period", "config"], as_index=False,).sum()
    gb_a.drop(labels="Subscription Term", axis=1, inplace=True)

    config_list = ["1Y", "2Y", "3Y", "MTHLY"]
    test1 = gb_a["config"].isin(config_list)

    gb_a_1Y = test1[test1["config"] == "1Y"]
    gb_a_2Y = test1[test1["config"] == "2Y"]
    gb_a_3Y = test1[test1["config"] == "3Y"]
    gb_a_1M = test1[test1["config"] == "MTHLY"]

    print("this is the lenght of type A 1M billings: ", len(gb_a_1M))
    print("this is the lenght of type A 1Y billings: ", len(gb_a_1Y))
    print("this is the lenght of type A 2Y billings: ", len(gb_a_2Y))
    print("this is the lenght of type A 3Y billings: ", len(gb_a_3Y))

    # Type D Billings
    dfr_d = dfr[dfr["rev_req_type"] == "D"]

    gb_d = dfr_d.groupby(["curr", "BU", "Period", "rebill_rule"], as_index=False,).sum()
    gb_d.drop(labels="Subscription Term", axis=1, inplace=True)

    gb_d_mthly = gb_d[gb_d["rebill_rule"].isin(["Y1", "Y2", "Y3", "Y5"])]
    gb_d_qtrly = gb_d[gb_d["rebill_rule"] == "YQ"]
    gb_d_four_mths = gb_d[gb_d["rebill_rule"] == "YT"]
    gb_d_semi_ann = gb_d[gb_d["rebill_rule"] == "YH"]
    gb_d_annual = gb_d[gb_d["rebill_rule"].isin(["YA", "YC"])]
    gb_d_two_yrs = gb_d[gb_d["rebill_rule"] == "Y"]

    print("Length of monthly", len(gb_d_mthly))
    print("Length of quarterly", len(gb_d_qtrly))
    print("Length of four months", len(gb_d_four_mths))
    print("Length of semi ann", len(gb_d_semi_ann))
    print("Length of annual", len(gb_d_annual))
    print("Length of two years", len(gb_d_two_yrs))

    # Recombining these dataframes into one large dataframe
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
    ]

    df = merge_all_dataframes(list_df, list_columns)

    df = clean_df_columns(df)

    return df


def merge_new_dataframe(old_df, new_df, new_column):
    df_merged = pd.merge(
        old_df,
        new_df,
        how="outer",
        left_on=["curr", "BU", "Period"],
        right_on=["curr", "BU", "Period"],
    )
    df_merged.rename(index=str, columns={"amount": new_column}, inplace=True)

    # need to drop the product configtype id for merges where the new_df is of type A
    config_str = "config"
    rule_str = "rebill_rule"
    if config_str in df_merged.columns:
        df_merged.drop(columns=["config"], inplace=True)

    if rule_str in df_merged.columns:
        df_merged.drop(columns=["rebill_rule"], inplace=True)

    return df_merged


def merge_all_dataframes(list_df, list_columns):
    for i, df in enumerate(list_df):
        print("This is i:", i)
        # print("This is the df: ", df.head())
        print("referencing the column: ", list_columns[i])

        if i == 0:
            df_merged = list_df[0]
            df_merged.rename(
                index=str, columns={"amount": list_columns[i]}, inplace=True
            )
        else:
            df_merged = merge_new_dataframe(df_merged, df, list_columns[i])

    return df_merged


def clean_df_columns(df):

    # clean up NaNs before adding
    df = df.fillna(value=0)

    # Monthly
    df["deferred_1M"] = df["deferred_1M_a"] + df["deferred_1M_d"]
    df.drop(labels=["deferred_1M_a", "deferred_1M_d"], axis=1, inplace=True)

    # Annual
    df["deferred_1Y"] = df["deferred_1Y_a"] + df["deferred_1Y_d"]
    df.drop(labels=["deferred_1Y_a", "deferred_1Y_d"], axis=1, inplace=True)

    # Two-Year
    df["deferred_2Y"] = df["deferred_2Y_a"] + df["deferred_2Y_d"]
    df.drop(labels=["deferred_2Y_a", "deferred_2Y_d"], axis=1, inplace=True)

    # renaming 3Y, 3M and 6M
    df.rename(
        index=str,
        columns={
            "deferred_3Y_a": "deferred_3Y",
            "deferred_3M_d": "deferred_3M",
            "deferred_6M_d": "deferred_6M",
        },
        inplace=True,
    )

    return df


df = clean_df_columns(df)

""" Now working on the ZCC billings """
df_cal = pd.read_excel("../data/old/ADOBE_FINANCIAL_CALENDAR.xlsx", "ADBE_cal")

df_ZCC = pd.read_excel("../data/old/type_D_ZCC_billings.xlsx", sheet_name="DC")

df_ZCC.rename(
    index=str,
    columns={
        "Document Currency": "curr",
        "Enterprise BU Description": "BU",
        "Rule for Bill Date Code": "rebill_rule",
        "Week of FICA Posting Date (YYYYMMDD) (copy)": "fiscal_week",
        "DF Additions - Doc Curr": "amount",
    },
    inplace=True,
)


ZCC_curr = df_ZCC["curr"].unique()

