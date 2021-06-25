"""
This file contains the functions used in the deferred revenue forecast
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

plt.style.use("ggplot")

def load_FX_data(config_dict):
    '''
    This function takes the config dictionary and creates a dataframe from the FX_data.xlsx file.
    This dataframe includes all of the currency exchange rates, their forward rates and vols

    :param config_dict: the main configuration dictionary for the deferred revenue model
    :return: df: The final dataframe contains the following fields
        DC - the 3 digit ticker for each document currency
        Ticker - the 6 digit currency ticker for the exchange rate to USD.
        Spot - the spot exchagne rate
        FWD_3M, FWD_6M, FWD_9M, FWD_1Y - the forward rates for the exchange rates
        VOL_3M, VOL_6M, VOL_9M, VOL_12M - the implied volatility for the exchange rates

    '''
    filename_FX = config_dict['path_to_data'] + config_dict['FX_rates']['filename']
    df = pd.read_excel(filename_FX, sheet_name=config_dict['FX_rates']['sheetname'])

    # some of the currencies do not contain forwards, so just filling across (as if no points)
    df = df.fillna(method='ffill', axis=1)

    # adjusting the vols to be in percentages
    df["VOL_3M"] = df["VOL_3M"] / 100
    df["VOL_6M"] = df["VOL_6M"] / 100
    df["VOL_9M"] = df["VOL_9M"] / 100
    df["VOL_1Y"] = df["VOL_1Y"] / 100

    return df


def load_curr_map(config_dict):
    '''
    load_curr_map creates a dataframe from the currency_map.xlsx file. This dataframe contains a list of
    countries and the currency that contains the largest percentage of billings in that country. This is later
    used to take the bookings forecast, which has country level information and mapping this to a document currency.
    :param config_dict: The main dictionary for the deferred revenue program
    :return: df: A dataframe containing the following columns
                - 'Country':
                - 'Currency':
    '''
    filename_curr_map = config_dict['path_to_data'] + config_dict['curr_map']['filename']
    curr_map_sheetname = config_dict['curr_map']['sheetname']
    df = pd.read_excel(filename_curr_map, sheet_name=curr_map_sheetname)

    # need to replace part of the strings in country that are in this report
    df["Country"] = df["Country"].str.replace(
        "\(MA\)", "", case=False
    )
    df["Country"] = df["Country"].str.strip()

    return df


def add_billings_periods(df_billings):
    '''
     - the billings dataframe does not contain every period if there are no bookings within a period.
     - the easiest way to create the forecast requires that we have all of the periods in each BU and currency pair (or at least 36 months worth so that we can incorporate the 3 year deferred bookings

      The bookings foreacast also contains products such as 'LiveCycle' and 'other solutions' that we do not expect to recieve billings for going forward (there are no booking associated with this) so we need to remove them from the billings data
      Explicit call to the add_billings_periods function is below

    :param df_billings: The main billings dataframe
    :return: df_billings: The main billings dataframe that now includes every period for every BU and currency

    '''
    # clean up billings by removing LiveCycle and other solutions
    index_lc = df_billings[df_billings["BU"] == "LiveCycle"].index
    df_billings.drop(index_lc, inplace=True)

    index_other = df_billings[df_billings["BU"] == "Other Solutions"].index
    df_billings.drop(index_other, inplace=True)

    all_BU = df_billings["BU"].unique()
    all_curr = df_billings["curr"].unique()

    all_periods = df_billings["period"].unique()
    all_periods = np.sort(all_periods)
    #all_periods = all_periods[-36:]
    all_periods = all_periods[-157:]

    list_new_BUs = []
    list_new_currs = []
    list_new_periods = []

    for this_BU in all_BU:

        for this_curr in all_curr:

            df_slice = df_billings[
                (df_billings["BU"] == this_BU) & (df_billings["curr"] == this_curr)
                ].copy()

            list_periods = df_slice["period"].unique()
            set_periods = set(list_periods)
            set_all = set(all_periods)

            periods_missing = set_all.difference(set_periods)

            for i in periods_missing:
                list_new_periods.append(i)
                list_new_currs.append(this_curr)
                list_new_BUs.append(this_BU)
    df_to_add = pd.DataFrame(
        {"curr": list_new_currs, "BU": list_new_BUs, "period": list_new_periods}
    )

    df_billings_check = pd.concat([df_billings, df_to_add], sort=False)

    df_billings_check = df_billings_check.fillna(0)

    df_billings = df_billings_check.copy()

    df_billings = df_billings.sort_values(
        ["curr", "BU", "period"], ascending=(True, True, True)
    )

    return df_billings


def load_ADBE_cal(config_dict):
    '''
    This function opens the Adobe financial calendar and creates a dataframe containing the information
    in the Adobe financial calendar that is needed for the deferred revenue model.
    The dataframe creates returns has two columns
        - "Period_Weeks: containing the number of weeks within a fiscal period
        - "period_match", which contains the period date in YYYY-MM format

    :param config_dict: the main configuration dictionary for the deferred revenue model
    :return: df: A dataframe containing all of the details of the financial calendar
    '''

    ADBE_cal_filename  = config_dict['ADBE_cal']['direct_filename']
    ADBE_cal_sheetname = config_dict['ADBE_cal']['sheetname']

    df = pd.read_excel(ADBE_cal_filename, ADBE_cal_sheetname)
    df["Period_Weeks"] = (df["Per_End"] - df["Per_Start"]) / np.timedelta64(
        1, "W"
    )
    df["Period_Weeks"] = df["Period_Weeks"].astype(int)
    df["Period_Weeks"] = df["Period_Weeks"] + 1

    # Creating a column in df_cal with year  '-' the last two digits of the per_ticker to match with the billings dataframe
    df["p2digit"] = df["Period"].astype(str)
    df["p2digit"] = df["p2digit"].str.zfill(2)

    df["period_match"] = (
            df["Year"].astype(str) + "-" + df["p2digit"].astype(str)
    )

    df.drop(["p2digit"], axis=1, inplace=True)

    df.drop(
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

    return df


def convert_fcst(df_fcst, df_FX_rates, list_columns, new_columns):
    for i in list_columns:
        new_column = i + "US"
        df_fcst[new_column] = 0

    # get the unique list of currency and BU combinations in the forecast
    v_un_BU, v_un_curr = find_unique_curr_and_BU(df_fcst)
    for i in range(len(v_un_BU)):
        this_BU = v_un_BU[i]
        this_curr = v_un_curr[i]
        print("working on BU: {0}  and currency: {1}".format(this_BU, this_curr))

        # create the list of forwards to use here
        these_forwards = df_FX_rates[df_FX_rates["DC"] == this_curr]
        just_forwards = these_forwards[new_columns]
        if these_forwards.is_direct.values == 1:

            transp_fwds = just_forwards.transpose(copy=True).values

        else:
            transp_fwds = just_forwards.transpose(copy=True).values
            transp_fwds = 1 / transp_fwds

        this_slice = df_fcst[
            (df_fcst["BU"] == this_BU) & (df_fcst["curr"] == this_curr)
            ].copy()

        # Debug trap
        if this_BU == "Print & Publishing":
            if this_curr == 'USD':
                print('we are here')
        for col in list_columns:
            new_column = col + "US"
            old_column = col + "DC"

            DC_values = this_slice[old_column].values
            DC_values = DC_values.reshape(-1, 1)
            transp_fwds = transp_fwds.reshape(-1, 1)

            if len(DC_values) != 12:
                print('DC values are not correct shape')
                print('Here are the DC_values')
                print(DC_values)
                print('Here is the shape', DC_values.shape)

            if len(transp_fwds) != 12:
                print('transp_fwds are not correct shape')
                print('Here are the transp_fwds')
                print(transp_fwds)
                print('Here is the shape', transp_fwds.shape)

            xx = DC_values * transp_fwds

            df_fcst.loc[
                (df_fcst["BU"] == this_BU) & (df_fcst["curr"] == this_curr), new_column
            ] = xx

    return df_fcst


def merge_bookings_to_fcst(df_book_period, df_fcst):

    #TODO: Fix this bomb here and put into config

    #Below clears out any periods and quarters that need to be dropped from the df_book_period file
    #If we are looking at a quarter in the future
    dc_list = ["P01_DC", "P02_DC", "P03_DC",
                "P04_DC", "P05_DC", "P06_DC",
                "P07_DC", "P08_DC", "P09_DC",
                "P10_DC", "P11_DC", "P12_DC"
               ]
    us_list = ["P01", "P02", "P03",
               "P04", "P05", "P06",
               "P07", "P08", "P09",
               "P10", "P11", "P12"]

    df_temp_book_period = df_book_period.drop(
        columns=[
            "Q1",
            "Q2",
            "Q3",
            "Q4",
            "FX_fwd_rate"
        ]
    ).copy()

    #df_temp_book_period = df_book_period.copy()
    df_DC = df_temp_book_period.copy()
    df_US = df_temp_book_period.copy()

    df_DC = df_DC.drop(columns=us_list)
    df_US = df_US.drop(columns=dc_list)

    df_DC_melt = pd.melt(df_DC, id_vars=["BU", "curr"])
    df_US_melt = pd.melt(df_US, id_vars=["BU", "curr"])

    df_DC_melt.rename(
        columns={"variable": "period", "value": "book_1Y_DC"}, inplace=True
    )
    df_US_melt.rename(
        columns={"variable": "period", "value": "book_1Y_US"}, inplace=True
    )

    df_DC_melt["period"] = df_DC_melt["period"].str.replace("P", "2021-")
    df_DC_melt["period"] = df_DC_melt["period"].str.replace("_DC", "")
    df_US_melt["period"] = df_US_melt["period"].str.replace("P", "2021-")
    df_US_melt["period"] = df_US_melt["period"].str.replace("_US", "")

    # reset index
    df_DC_melt.set_index(["BU", "curr", "period"], inplace=True)

    df_US_melt.set_index(["BU", "curr", "period"], inplace=True)

    df_melted = df_DC_melt.join(df_US_melt, how="left")

    df_fcst.set_index(["BU", "curr", "period"], inplace=True)
    print('Below is the df_fcst')
    print(df_fcst.head(10))

    print('Below is the df_melted dataframe')
    print(df_melted.head(10))

    df_fcst = df_fcst.join(df_melted, how="left")
    df_fcst.fillna(0, inplace=True)
    df_fcst.reset_index(inplace=True)
    return df_fcst


def bring_slice_wf_forward(df):
    wf_cols_copy = df.columns[df.columns.str.contains("P")]
    print(wf_cols_copy)
    df = df.reset_index(drop=True)
    for index, row in df.iterrows():

        if index < len(df) - 1:
            this_row = df.loc[index, wf_cols_copy].to_numpy()
            this_row = np.delete(this_row, 0)
            this_row = np.append(this_row, [0])
            df.loc[index + 1, wf_cols_copy] += this_row

    return df


def bring_initial_wf_forward(df_waterfall):
    list_BU = df_waterfall["BU"].unique()

    for i in range(len(list_BU)):
        this_BU = list_BU[i]
        this_slice = df_waterfall[df_waterfall["BU"] == this_BU].copy()

        df_this_wf = bring_slice_wf_forward(this_slice)

        if i == 0:
            df_wf = df_this_wf.copy()
        else:
            df_wf = pd.concat([df_wf, df_this_wf], sort=False)

    df_wf.reset_index(drop=True, inplace=True)
    return df_wf


def sum_USD_amt(list_df, list_columns):
    total_US = []
    for df in list_df:
        total_US.append(df["US_amount"].sum())
    total_df = pd.DataFrame(index=list_columns, columns=["US_amounts"], data=total_US)
    return total_df


def merge_all_dataframes(list_df, list_columns):
    '''
    This function takes a list of dataframes and a list of column names and creates a single dataframe
    containing all of the dataframes with column names contained in list_columns.

    It is used to take the grouped dataframes in list_df that are grouped by their rebill frequency
    and map them to a new column that contains their rebill requency in the column name.

    :param list_df: The list of dataframes created in the load_base_billings file
    :param list_columns: The column names representing the type of billing and frequency of rebilling

    :return: df: This is the main dataframe for the billings we have classified. The dataframe contains the following columns
                'curr': The three digit currency for the row
                'BU': The Enterprise BU for the billing
                'period': The period of the billing in YYYY-PP format
                'recognized_DC': The recognized document currency billings that go immediately to revenue
                'recognized_US': The USD equivalent of these immediate revenue billings
                'service_DC': The document currency billings for service billings that get placed into deferred and are
                            moved to revenue as service is performed
                'service_US': The USD equivalent of the service billings
                'deferred_1M_DC': document currency deferred billings that will renew every month
                'deferred_1M_US': USD equivalent of deferred 1M billings
                'deferred_3M_DC': document currency deferred billings that will renew every 3 months
                'deferred_3M_US': USD equivalent of deferred 3M billings
                'deferred_6M_DC':document currency deferred billings that will renew every 6 months
                'deferred_6M_US': USD equivalent of deferred 6M billings
                'deferred_1Y_DC': document currency deferred billings that will renew every year
                'deferred_1Y_US': USD equivalent of deferred annual billings
                'deferred_2Y_DC': document currency deferred billings that will renew every 2 years
                'deferred_2Y_US': USD equivalent of deferred 2Y billings
                'deferred_3Y_DC': document currency deferred billings that will renew every 3 year
                'deferred_3Y_US': USD equivalent of deferred 3Y billings

    '''
    for i, df in enumerate(list_df):
        # print('This is i:', i)
        # print('referencing the column: ', list_columns[i])

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


def merge_new_dataframe(old_df, new_df, new_column):
    '''
    This program is called in the 'merge_all_dataframes' function and does the actual act of merging the dataframes
    :param old_df: The initial dataframe
    :param new_df: The new dataframe to be merged
    :param new_column: The name of the new column to be added to the dataframe representing the type of
                        deferred billing and the frequency of the billing
    :return: the merged dataframe
    '''
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


def clean_df_columns(df):
    '''
    This function cleans up the dataframe columns that get created when we merge all of the dataframes
    in the load_base_billings file. The dataframe columns contain billings from both type A and type D
    billings that are the same rebillings frequency. We don;t care if they are type A or type D at this
    point in the program because this designation was only helpful to determine the frequency of the billings.
    So here we merge these columns to be together.

    The final columns contain information about (up to) 3 categories.
        1. Type of billings: servivces, recognized, deferred
        2. Frequency of rebillings: 1M, 3M, 6M, 1Y, 2Y, 3Y (for deferred billings)
        3. Document currency or USD: DC or USD

    :param df: The initial merged dataframe created in merge_all_dataframes
    :return: df: a dataframe with the following columns
                'curr': The three digit currency for the row
                'BU': The Enterprise BU for the billing
                'period': The period of the billing in YYYY-PP format
                'recognized_DC': The recognized document currency billings that go immediately to revenue
                'recognized_US': The USD equivalent of these immediate revenue billings
                'service_DC': The document currency billings for service billings that get placed into deferred and are
                            moved to revenue as service is performed
                'service_US': The USD equivalent of the service billings
                'deferred_1M_DC': document currency deferred billings that will renew every month
                'deferred_1M_US': USD equivalent of deferred 1M billings
                'deferred_3M_DC': document currency deferred billings that will renew every 3 months
                'deferred_3M_US': USD equivalent of deferred 3M billings
                'deferred_6M_DC':document currency deferred billings that will renew every 6 months
                'deferred_6M_US': USD equivalent of deferred 6M billings
                'deferred_1Y_DC': document currency deferred billings that will renew every year
                'deferred_1Y_US': USD equivalent of deferred annual billings
                'deferred_2Y_DC': document currency deferred billings that will renew every 2 years
                'deferred_2Y_US': USD equivalent of deferred 2Y billings
                'deferred_3Y_DC': document currency deferred billings that will renew every 3 year
                'deferred_3Y_US': USD equivalent of deferred 3Y billings

    '''
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
    #df.rename(
    #    index=str,
    #    columns={"curr": "curr", "BU": "BU", "period": "period"},
    #    inplace=True,
    #)

    return df


def remove_bad_currencies(df, model_dict):
    this_list = df["curr"].unique().tolist()

    for curr in model_dict["curr_removed"]:
        if curr in this_list:
            print("Removed the following currency: ", curr)
            df = df[df["curr"] != curr]
    return df


def load_FX_fwds(config_dict):
    '''
    load_FX_fwds takes the config dictionary and returns a dataframe that contains the foreign exchange
    forward rates that FP&A uses in their financial plan. These forward rates are later used to determine
    document currency billings from the bookings forecast, which is USD based.

    :param config_dict: the main configuration dictionary for the deferred revenue program
    :return: df: A dataframe consisting of
                - 'curr': The 3 digit ticker for a currency
                - 'forward': The forward rate used by FP&A for bookings forecasts
    '''
    filename_FX_fwds = config_dict['path_to_data'] + config_dict['FX_forwards']['filename']
    FX_fwds_sheetname  = config_dict['FX_forwards']['sheetname']
    df = pd.read_excel(
        filename_FX_fwds, sheet_name=FX_fwds_sheetname, skiprows=1, usecols="C,G",
    )

    df.rename(
        index=str, columns={"Unnamed: 2": "curr", "FWD REF": "forward"}, inplace=True
    )
    return df


def find_unique_curr_and_BU(df_billings):
    v_BU = df_billings["BU"].copy()
    v_curr = df_billings["curr"].copy()
    v_both = v_BU + v_curr
    v_unique = v_both.unique()

    v_un_BU = [sub[:-3] for sub in v_unique]
    v_un_curr = [sub[-3:] for sub in v_unique]

    return v_un_BU, v_un_curr


def create_billing_forecast(df_billings, df_fcst):
    v_un_BU, v_un_curr = find_unique_curr_and_BU(df_billings)

    print('Debug Stop')
    # new Vectorized approach (sort of)
    counter = 0

    for i in range(len(v_un_BU)):
        this_BU = v_un_BU[i]
        this_curr = v_un_curr[i]

        print("working on BU: {0}  and currency: {1}".format(this_BU, this_curr))
        df_slice = df_billings[
            (df_billings["BU"] == this_BU) & (df_billings["curr"] == this_curr)
            ].copy()
        list_bill_periods = df_billings["period"].unique()
        list_bill_periods.sort()

        old_per_3Y = list_bill_periods[-36:-24]
        old_per_2Y = list_bill_periods[-24:-12]
        old_per_1Y = list_bill_periods[-12:]
        old_per_6M = list_bill_periods[-6:]
        old_per_3M = list_bill_periods[-3:]

        # three year
        this_v_3yrs = df_slice.loc[
            df_slice["period"].isin(old_per_3Y), "deferred_3Y_DC"
        ].copy()
        if len(this_v_3yrs) != 12:
            print(this_BU, this_curr)
            print(
                "There is a period mismatch. length of 3yrs vector = ", len(this_v_3yrs)
            )
            print("Length of df_slice: ", len(df_slice))
            print("This BU: {0} and this currency: {1}".format(this_BU, this_curr))

        else:
            df_fcst.loc[
                (df_fcst["BU"] == this_BU) & (df_fcst["curr"] == this_curr),
                "deferred_3Y_DC",
            ] = this_v_3yrs.values

        # two years
        this_v_2yrs = df_slice.loc[
            df_slice["period"].isin(old_per_2Y), "deferred_2Y_DC"
        ].copy()
        if len(this_v_2yrs) != 12:
            print(this_BU, this_curr)
            print(
                "There is a period mismatch. length of 2 yrs vector = ",
                len(this_v_2yrs),
            )
            print("Length of df_slice: ", len(df_slice))
            print("This BU: {0} and this currency: {1}".format(this_BU, this_curr))
        else:
            df_fcst.loc[
                (df_fcst["BU"] == this_BU) & (df_fcst["curr"] == this_curr),
                "deferred_2Y_DC",
            ] = this_v_2yrs.values

        # one year
        this_v_1yrs = df_slice.loc[
            df_slice["period"].isin(old_per_1Y), "deferred_1Y_DC"
        ].copy()
        if len(this_v_1yrs) != 12:
            print(this_BU, this_curr)
            print(
                "There is a period mismatch. length of 1 yr vector = ", len(this_v_1yrs)
            )
            print("Length of df_slice: ", len(df_slice))

        else:
            df_fcst.loc[
                (df_fcst["BU"] == this_BU) & (df_fcst["curr"] == this_curr),
                "deferred_1Y_DC",
            ] = this_v_1yrs.values

        # six months (we need to append the values to repeat once)
        this_v_6M = df_slice.loc[
            df_slice["period"].isin(old_per_6M), "deferred_6M_DC"
        ].copy()
        this_v_6M = this_v_6M.append(this_v_6M, ignore_index=True)

        df_fcst.loc[
            (df_fcst["BU"] == this_BU) & (df_fcst["curr"] == this_curr),
            "deferred_6M_DC",
        ] = this_v_6M.values

        # three months:
        this_v_3M = df_slice.loc[
            df_slice["period"].isin(old_per_3M), "deferred_3M_DC"
        ].copy()
        this_v_3M = this_v_3M.append(this_v_3M, ignore_index=True)
        this_v_3M = this_v_3M.append(this_v_3M, ignore_index=True)

        df_fcst.loc[
            (df_fcst["BU"] == this_BU) & (df_fcst["curr"] == this_curr),
            "deferred_3M_DC",
        ] = this_v_3M.values

        # what the hell do we do with the service and recognized revenue billings?
        # RECOGNIZED REVENUE - does not go to deferred, so just take the last 12 month's worth
        this_recog = df_slice.loc[
            df_slice["period"].isin(old_per_1Y), "recognized_DC"
        ].copy()
        df_fcst.loc[
            (df_fcst["BU"] == this_BU) & (df_fcst["curr"] == this_curr), "recognized_DC"
        ] = this_recog.values

        # SERVICE BASED BILLINGS - for now just use the average of whatever we used last time
        this_svc = df_slice.loc[
            df_slice["period"].isin(old_per_1Y), "service_DC"
        ].copy()
        df_fcst.loc[
            (df_fcst["BU"] == this_BU) & (df_fcst["curr"] == this_curr), "service_DC"
        ] = this_svc.values

        # Type B Deferred (Service Billings)
        #this_type_B = df_slice.loc[
        #    df_slice["period"].isin(old_per_1Y), "deferred_B_DC"
        #].copy()
        #df_fcst.loc[
        #    (df_fcst["BU"] == this_BU) & (df_fcst["curr"] == this_curr), "deferred_B_DC"
        #] = this_type_B.values

        # MONTHLY BILLINGS
        # here we need to call a seperate function using just the X array that is the one month billings
        this_y = df_slice["deferred_1M_DC"].copy()
        this_y = this_y.to_numpy()
        this_y = this_y.reshape(-1, 1)

        if sum(this_y) != 0:
            period_weeks = df_slice["Period_Weeks"].copy()
            period_weeks = period_weeks.to_numpy()
            period_weeks = period_weeks.reshape(-1, 1)

            this_y = np.true_divide(this_y, period_weeks, where=period_weeks>0)
            this_y = np.nan_to_num(this_y)
            X = np.arange(len(this_y))

            this_model = build_monthly_forecast(X, this_y)
            weekly_fcst_y = this_model["fcst_y"]

            fcst_slice = df_fcst[
                (df_fcst["BU"] == this_BU) & (df_fcst["curr"] == this_curr)
                ].copy()
            fcst_weeks = fcst_slice["Period_Weeks"].to_numpy()
            fcst_weeks = fcst_weeks.reshape(-1, 1)

            period_fcst_y = weekly_fcst_y * fcst_weeks

            # print('length of new_y: ', len(fcst_y))
            df_fcst.loc[
                (df_fcst["BU"] == this_BU) & (df_fcst["curr"] == this_curr),
                "deferred_1M_DC",
            ] = period_fcst_y

            df_fcst.loc[
                (df_fcst["BU"] == this_BU) & (df_fcst["curr"] == this_curr), "r_squared"
            ] = this_model["score"]

            df_fcst.loc[
                (df_fcst["BU"] == this_BU) & (df_fcst["curr"] == this_curr), "intercept"
            ] = this_model["intercept"][0]

            df_fcst.loc[
                (df_fcst["BU"] == this_BU) & (df_fcst["curr"] == this_curr), "coeff"
            ] = this_model["coeff"][0]

            df_fcst.loc[
                (df_fcst["BU"] == this_BU) & (df_fcst["curr"] == this_curr), "X_length"
            ] = this_model["first_row"]

        # print('For this BU: {0} and this currency {1}, we have a score of {2}, and int of {3} and a coeff of {4}'.
        #     format(this_BU, this_curr, this_score, this_int, this_coeff))
        # NOTE: We will need to return two things here
        # First - the df_fcst dataframe
        # second - a dictionary describing the monthly forecasts

    return df_fcst


def build_monthly_forecast(X, y):
    """
    Need to keep track of the initial X and Y
    Need to track the best X, best Y, best model & best score
    Within the loop, reducing the X and Y to new_x and new_y and keeping track of new model
    If the new model is better, the best_x, best_y, best_model and best_score are overwritten

    At the end of the program, the best model is fit and the relevant information is returned
    """

    X = X.reshape(-1, 1)
    y = y.reshape(-1, 1)

    best_X = X.reshape(-1, 1)
    best_y = y.reshape(-1, 1)

    fcst_X = np.arange(np.max(X) + 1, np.max(X) + 13)
    fcst_X = fcst_X.reshape(-1, 1)

    # best row tracks the beginning month used for the model
    best_row = 0

    # create initial linear regression model, fit it and record score
    best_model = LinearRegression(fit_intercept=True)
    best_model.fit(best_X, best_y)
    best_score = best_model.score(best_X, best_y)
    best_int = best_model.intercept_
    best_coeff = best_model.coef_

    print("Model Score :",       best_score)
    print("Model intercept :",   best_model.intercept_)
    print("Model Coefficient :", best_model.coef_)

    for start_row in np.arange(1, y.shape[0] - 12):
        new_X = X[start_row:]
        new_y = y[start_row:]

        new_model = LinearRegression(fit_intercept=True)
        new_model.fit(new_X, new_y)
        new_score = new_model.score(new_X, new_y)
        new_int = new_model.intercept_
        new_coeff = new_model.coef_

        print("Model Score :",       new_score)
        print("Model intercept :",   new_model.intercept_)
        print("Model Coefficient :", new_model.coef_)

        # if the new model beats the best model, reassign to the best model
        if new_score > best_score:
            best_model = new_model
            best_score = new_score
            best_X = new_X
            best_y = new_y
            best_row = start_row
            best_int = new_int
            best_coeff = new_coeff

    # perform the forecast
    fcst_y = best_model.predict(fcst_X)

    # ADDED 12/7/2020 due to error adding coeff to the dataframe
    best_coeff = best_coeff.flatten()
    monthly_model = dict(
        {
            "model": best_model,
            "score": best_score,
            "fcst_y": fcst_y,
            "first_row": best_row,
            "intercept": best_int,
            "coeff": best_coeff,
        }
    )

    return monthly_model

def load_bookings(config_dict):
    '''
    The bookings files come from FP&A and the structure of these files changes often

    :param config_dict:
    :return: df_bookings
    '''
    filename_DX = config_dict['path_to_data'] + config_dict['bookings']['filename_DX']
    sheetname_DX = config_dict['bookings']['sheetname_DX']
    start_row_DX = config_dict['bookings']['start_row_DX']

    filename_DME = config_dict['path_to_data'] + config_dict['bookings']['filename_DME']
    sheetname_DME = config_dict['bookings']['sheetname_DME']


    df_DX = load_DX_bookings(filename_DX, sheetname_DX, start_row_DX)
    df_DME = load_DME_bookings(filename_DME, sheetname_DME)
    df = pd.concat([df_DME, df_DX])

    # Now we need to melt the dataframe so that the columns for each quarter are in one row
    df = pd.melt(df, id_vars = ['BU', 'geo', 'region', 'country'],
                 value_vars = ['Q1_2021', 'Q2_2021', 'Q3_2021', 'Q4_2021'],
                 var_name = 'Quarter')

    # Extra white space in the FP&A files
    df['BU'] = df['BU'].str.strip()
    #df['segment'] = df['segment'].str.strip()
    df['geo'] = df['geo'].str.strip()
    df['region'] = df['region'].str.strip()
    df['country'] = df['country'].str.strip()
    return df

def load_DME_bookings(filename, sheetname):

    df_DME = pd.read_excel(filename, sheetname)

    df_DME = df_DME.rename(columns={'Metrics': 'metrics',
                            'Profit center': 'profit_center',
                            'Market Area': 'market_area',
                            'Market Segement': 'segment',
                            'Q1 2021':'Q1_2021',
                            'Q2 2021':'Q2_2021',
                            'Q3 2021':'Q3_2021',
                            'Q4 2021':'Q4_2021'
                            })

    # Drop the columns that we will not use
    df_DME = df_DME.drop(columns = ['segment', 'GTM', '2021'])

    # only want the Net ACV bookings
    df_DME = df_DME[df_DME['metrics']=='Net ACV']

    # ADJUSTING THE profit_center to not double count numbers
    # creating the BU_ID
    df_DME['BU_id'] =  df_DME['profit_center'].apply(lambda st: st[0:st.find("-")])
    df_DME['BU_segment'] = df_DME['profit_center'].apply(lambda st: st[st.find("-")+1:])

    df_DME['BU_id'] = df_DME['BU_id'].str.strip()
    df_DME['BU_segment'] = df_DME['BU_segment'].str.strip()

    list_BU_keepers = ['EB10', 'EB15']
    df_DME = df_DME[df_DME['BU_id'].isin(list_BU_keepers)]

    # NOW WORKING ON GEO, REGION and MARKET AREA
    # identify the characters in a string
    df_DME['in_parens'] =  df_DME['market_area'].apply(lambda st: st[st.find("(")+1:st.find(")")])

    df_DME['market_area'] = df_DME['market_area'].apply(lambda st: st[0:st.find("(")-1])

    df_DME['pc_ID'] = df_DME['profit_center'].apply(lambda st: st[0:st.find('-')])
    df_DME['pc_descr'] = df_DME['profit_center'].apply(lambda st: st[st.find('-')+1:])

    df_DME['geo'] = df_DME[df_DME['in_parens']=='G']['market_area']
    df_DME['geo'] =df_DME['geo'].ffill()

    df_DME['region'] = df_DME[df_DME['in_parens']=='R']['market_area']
    df_DME['region'] = df_DME['region'].ffill()

    # filter to just include market area
    df_DME = df_DME[df_DME['in_parens']=='MA'].copy()

    # drop unnecessary columns and reorder the columns
    df_DME = df_DME.drop(columns=['profit_center', 'in_parens' ])

    # Rename pc_descr to be BU
    ### THIS WAS THE ERROR. it was renamed to segment
    df_DME.rename(columns = {'pc_descr': 'BU',
                             'market_area':'country'}, inplace=True)

    # Add the BU
    #df_DME['BU'] = 'Digital Media'

    # We need to remove the segment data: it is not included in the DME bookings
    df_DME = df_DME.groupby(by = ['BU', 'geo', 'region', 'country']).sum()
    df_DME = df_DME.reset_index()

    df_DME = df_DME[['BU', 'geo', 'region', 'country', 'Q1_2021','Q2_2021', 'Q3_2021', 'Q4_2021']]


    print('Done with the DME dataframe:')
    print(df_DME.sum())

    return df_DME


def load_DX_bookings(filename, sheetname, start_row):
    df_DX = pd.read_excel(filename, sheetname, skiprows=start_row)


    df_DX = df_DX.rename(columns = {'Unnamed: 0': 'segment',
                                    'Unnamed: 1': 'market_area',
                                    'Unnamed: 2': 'profit_center',
                                    'Q1 2021':'Q1_2021',
                                    'Q2 2021':'Q2_2021',
                                    'Q3 2021':'Q3_2021',
                                    'Q4 2021':'Q4_2021'})

    df_DX = df_DX.drop(columns = ['segment', '2021'])

    # identify the characters in a string
    df_DX['in_parens'] =  df_DX['market_area'].apply(lambda st: st[st.find("(")+1:st.find(")")])

    df_DX['market_area'] = df_DX['market_area'].apply(lambda st: st[0:st.find("(")-1])

    df_DX['pc_ID'] = df_DX['profit_center'].apply(lambda st: st[0:st.find('-')])
    df_DX['pc_descr'] = df_DX['profit_center'].apply(lambda st: st[st.find('-')+1:])

    df_DX['geo'] = df_DX[df_DX['in_parens']=='G']['market_area']
    df_DX['geo'] =df_DX['geo'].ffill()

    df_DX['region'] = df_DX[df_DX['in_parens']=='R']['market_area']
    df_DX['region'] = df_DX['region'].ffill()

    # filter to just include market area
    df_DX = df_DX[df_DX['in_parens']=='MA'].copy()


    # Adding BU and Segment information (all Exp Cloud)
    # THIS WAS WHERE THE ERROR WAS
    #df_DX['segment'] = 'Digital Experience'
    df_DX['BU'] = 'Experience Cloud'

    # drop unnecessary columns and reorder the columns
    df_DX = df_DX[['BU', 'geo', 'region', 'market_area', 'Q1_2021','Q2_2021', 'Q3_2021', 'Q4_2021']]

    # rename the market_area to be country
    df_DX.rename(columns={'market_area': 'country'}, inplace=True)

    # We need to remove the segment data: it is not included in the DME bookings
    df_DX = df_DX.groupby(by = ['BU', 'geo', 'region', 'country']).sum()
    df_DX = df_DX.reset_index()

    print('Done with the DX dataframe:')
    print(df_DX.sum())

    return df_DX


def build_deferred_waterfall(df_billings):
    '''
    This function takes the df_billings file and creates a deferred revenue waterfall for the
    forecasted billings.
    The columns contain the amortization by period ("p_1" to "p_36")
    The rows are grouped by period and BU.

    Other assumptions
     - Monthly Deferred Billings
    These occur in the middle of the month. Half the billings go directly to revenue,
    the remainder amortize out of deferred the next month

    - Three Month Deferred Billings
    These are assumed to occur at the end of the period.

    - Annual Billings
    1/12 of the current annual billings + 11 of the last annual billings + 1/12 of the year prior billings

    :param df_billings:
    :return: df_waterfall
    '''
    # Finding the unique currencies and BUs to slice the dataframe and build a waterfall for each
    v_un_BU, v_un_curr = find_unique_curr_and_BU(df_billings)

    # creating the waterfall list of numeric columns
    wf_columns = ["Total"]
    for i in np.arange(36):
        this_column = "p_" + str(i + 1)
        wf_columns.append(this_column)

    # creating the loop for the individual BU/curr waterfalls
    for i in range(len(v_un_BU)):
        this_BU = v_un_BU[i]
        this_curr = v_un_curr[i]

        print("working on BU: {0}  and currency: {1}".format(this_BU, this_curr))
        this_slice = df_billings[
            (df_billings["BU"] == this_BU) & (df_billings["curr"] == this_curr)
            ].copy()

        df_this_wf = this_slice[["curr", "BU", "period"]].copy()
        for item in wf_columns:
            df_this_wf[item] = 0

        df_this_wf = build_deferred_waterfall_slice(df_this_wf, this_slice)
        df_this_wf = bring_wf_forward(df_this_wf)

        if i == 0:
            df_waterfall = df_this_wf.copy()
        else:
            df_waterfall = pd.concat([df_waterfall, df_this_wf], sort=False)

    df_waterfall.reset_index(drop=True, inplace=True)

    return df_waterfall


def build_deferred_waterfall_slice(df_this_wf, this_slice):
    # Need to add half to the revenue piece
    df_this_wf["p_1"] += this_slice["deferred_1M_US"] * 0.5

    # 1/6 goes to revenue in the period it is billed
    df_this_wf["p_1"] += this_slice["deferred_3M_US"] * (1 / 3)
    df_this_wf["p_2"] += this_slice["deferred_3M_US"] * (1 / 3)
    df_this_wf["p_3"] += this_slice["deferred_3M_US"] * (1 / 6)

    # 1/12th directly to revenue
    df_this_wf["p_1"] += this_slice["deferred_6M_US"] * (1 / 6)
    df_this_wf["p_2"] += this_slice["deferred_6M_US"] * (1 / 6)
    df_this_wf["p_3"] += this_slice["deferred_6M_US"] * (1 / 6)
    df_this_wf["p_4"] += this_slice["deferred_6M_US"] * (1 / 6)
    df_this_wf["p_5"] += this_slice["deferred_6M_US"] * (1 / 6)
    df_this_wf["p_6"] += this_slice["deferred_6M_US"] * (1 / 12)

    # 1/24th directly to revenue
    df_this_wf["p_1"] += this_slice["deferred_1Y_US"] * (1 / 12)
    df_this_wf["p_2"] += this_slice["deferred_1Y_US"] * (1 / 12)
    df_this_wf["p_3"] += this_slice["deferred_1Y_US"] * (1 / 12)
    df_this_wf["p_4"] += this_slice["deferred_1Y_US"] * (1 / 12)
    df_this_wf["p_5"] += this_slice["deferred_1Y_US"] * (1 / 12)
    df_this_wf["p_6"] += this_slice["deferred_1Y_US"] * (1 / 12)
    df_this_wf["p_7"] += this_slice["deferred_1Y_US"] * (1 / 12)
    df_this_wf["p_8"] += this_slice["deferred_1Y_US"] * (1 / 12)
    df_this_wf["p_9"] += this_slice["deferred_1Y_US"] * (1 / 12)
    df_this_wf["p_10"] += this_slice["deferred_1Y_US"] * (1 / 12)
    df_this_wf["p_11"] += this_slice["deferred_1Y_US"] * (1 / 12)
    df_this_wf["p_12"] += this_slice["deferred_1Y_US"] * (1 / 24)

    # 1/24th 1 year bookings
    df_this_wf["p_1"] += this_slice["book_1Y_US"] * (1 / 12)
    df_this_wf["p_2"] += this_slice["book_1Y_US"] * (1 / 12)
    df_this_wf["p_3"] += this_slice["book_1Y_US"] * (1 / 12)
    df_this_wf["p_4"] += this_slice["book_1Y_US"] * (1 / 12)
    df_this_wf["p_5"] += this_slice["book_1Y_US"] * (1 / 12)
    df_this_wf["p_6"] += this_slice["book_1Y_US"] * (1 / 12)
    df_this_wf["p_7"] += this_slice["book_1Y_US"] * (1 / 12)
    df_this_wf["p_8"] += this_slice["book_1Y_US"] * (1 / 12)
    df_this_wf["p_9"] += this_slice["book_1Y_US"] * (1 / 12)
    df_this_wf["p_10"] += this_slice["book_1Y_US"] * (1 / 12)
    df_this_wf["p_11"] += this_slice["book_1Y_US"] * (1 / 12)
    df_this_wf["p_12"] += this_slice["book_1Y_US"] * (1 / 24)

    # Two year
    # 1/48th directly to revenue
    df_this_wf["p_1"] += this_slice["deferred_2Y_US"] * (1 / 24)
    df_this_wf["p_2"] += this_slice["deferred_2Y_US"] * (1 / 24)
    df_this_wf["p_3"] += this_slice["deferred_2Y_US"] * (1 / 24)
    df_this_wf["p_4"] += this_slice["deferred_2Y_US"] * (1 / 24)
    df_this_wf["p_5"] += this_slice["deferred_2Y_US"] * (1 / 24)
    df_this_wf["p_6"] += this_slice["deferred_2Y_US"] * (1 / 24)
    df_this_wf["p_7"] += this_slice["deferred_2Y_US"] * (1 / 24)
    df_this_wf["p_8"] += this_slice["deferred_2Y_US"] * (1 / 24)
    df_this_wf["p_9"] += this_slice["deferred_2Y_US"] * (1 / 24)
    df_this_wf["p_10"] += this_slice["deferred_2Y_US"] * (1 / 24)
    df_this_wf["p_11"] += this_slice["deferred_2Y_US"] * (1 / 24)
    df_this_wf["p_12"] += this_slice["deferred_2Y_US"] * (1 / 24)
    df_this_wf["p_13"] += this_slice["deferred_2Y_US"] * (1 / 24)
    df_this_wf["p_14"] += this_slice["deferred_2Y_US"] * (1 / 24)
    df_this_wf["p_15"] += this_slice["deferred_2Y_US"] * (1 / 24)
    df_this_wf["p_16"] += this_slice["deferred_2Y_US"] * (1 / 24)
    df_this_wf["p_17"] += this_slice["deferred_2Y_US"] * (1 / 24)
    df_this_wf["p_18"] += this_slice["deferred_2Y_US"] * (1 / 24)
    df_this_wf["p_19"] += this_slice["deferred_2Y_US"] * (1 / 24)
    df_this_wf["p_20"] += this_slice["deferred_2Y_US"] * (1 / 24)
    df_this_wf["p_21"] += this_slice["deferred_2Y_US"] * (1 / 24)
    df_this_wf["p_22"] += this_slice["deferred_2Y_US"] * (1 / 24)
    df_this_wf["p_23"] += this_slice["deferred_2Y_US"] * (1 / 24)
    df_this_wf["p_24"] += this_slice["deferred_2Y_US"] * (1 / 48)

    # Three year
    # 1/72nth directly to revenue
    df_this_wf["p_1"] += this_slice["deferred_3Y_US"] * (1 / 36)
    df_this_wf["p_2"] += this_slice["deferred_3Y_US"] * (1 / 36)
    df_this_wf["p_3"] += this_slice["deferred_3Y_US"] * (1 / 36)
    df_this_wf["p_4"] += this_slice["deferred_3Y_US"] * (1 / 36)
    df_this_wf["p_5"] += this_slice["deferred_3Y_US"] * (1 / 36)
    df_this_wf["p_6"] += this_slice["deferred_3Y_US"] * (1 / 36)
    df_this_wf["p_7"] += this_slice["deferred_3Y_US"] * (1 / 36)
    df_this_wf["p_8"] += this_slice["deferred_3Y_US"] * (1 / 36)
    df_this_wf["p_9"] += this_slice["deferred_3Y_US"] * (1 / 36)
    df_this_wf["p_10"] += this_slice["deferred_3Y_US"] * (1 / 36)
    df_this_wf["p_11"] += this_slice["deferred_3Y_US"] * (1 / 36)
    df_this_wf["p_12"] += this_slice["deferred_3Y_US"] * (1 / 36)
    df_this_wf["p_13"] += this_slice["deferred_3Y_US"] * (1 / 36)
    df_this_wf["p_14"] += this_slice["deferred_3Y_US"] * (1 / 36)
    df_this_wf["p_15"] += this_slice["deferred_3Y_US"] * (1 / 36)
    df_this_wf["p_16"] += this_slice["deferred_3Y_US"] * (1 / 36)
    df_this_wf["p_17"] += this_slice["deferred_3Y_US"] * (1 / 36)
    df_this_wf["p_18"] += this_slice["deferred_3Y_US"] * (1 / 36)
    df_this_wf["p_19"] += this_slice["deferred_3Y_US"] * (1 / 36)
    df_this_wf["p_20"] += this_slice["deferred_3Y_US"] * (1 / 36)
    df_this_wf["p_21"] += this_slice["deferred_3Y_US"] * (1 / 36)
    df_this_wf["p_22"] += this_slice["deferred_3Y_US"] * (1 / 36)
    df_this_wf["p_23"] += this_slice["deferred_3Y_US"] * (1 / 36)
    df_this_wf["p_24"] += this_slice["deferred_3Y_US"] * (1 / 36)
    df_this_wf["p_25"] += this_slice["deferred_3Y_US"] * (1 / 36)
    df_this_wf["p_26"] += this_slice["deferred_3Y_US"] * (1 / 36)
    df_this_wf["p_27"] += this_slice["deferred_3Y_US"] * (1 / 36)
    df_this_wf["p_28"] += this_slice["deferred_3Y_US"] * (1 / 36)
    df_this_wf["p_29"] += this_slice["deferred_3Y_US"] * (1 / 36)
    df_this_wf["p_30"] += this_slice["deferred_3Y_US"] * (1 / 36)
    df_this_wf["p_31"] += this_slice["deferred_3Y_US"] * (1 / 36)
    df_this_wf["p_32"] += this_slice["deferred_3Y_US"] * (1 / 36)
    df_this_wf["p_33"] += this_slice["deferred_3Y_US"] * (1 / 36)
    df_this_wf["p_34"] += this_slice["deferred_3Y_US"] * (1 / 36)
    df_this_wf["p_35"] += this_slice["deferred_3Y_US"] * (1 / 36)
    df_this_wf["p_36"] += this_slice["deferred_3Y_US"] * (1 / 72)

    return df_this_wf


def bring_wf_forward(df):
    wf_cols_copy = df.columns.str.contains("p_")

    df = df.reset_index(drop=True)
    for index, row in df.iterrows():

        if index < len(df) - 1:
            this_row = df.loc[index, wf_cols_copy].to_numpy()
            this_row = np.delete(this_row, 0)
            this_row = np.append(this_row, [0])
            df.loc[index + 1, wf_cols_copy] += this_row

    return df


def add_type_A_billings(config_dict, df, model_dict):
    '''
    Type A billings that do not have a valid product_config type cannot be processed using the base_billings query
    from tableau. To classify these billings, we need to look at deferred revenue additions (versus billings) that are
    type_A and have an invalid product_config type. The 'all_billings_inputs.xlsx' file contains a tab
    'type_A_no_config' that contains the following columns for these billings
        Document Currency - the document currency of the billings
        Enterprise BU Desc - BU
        Invoice Fiscal Year Period Desc - the period of the billings
        Product Mgmt BU - not used (product)
        Rev REc Contract End Date Hdr - one of the 2 end date fields
        Rev REc Contract End Date Item- one of the 2 end date fields
        Rev REc Contract Start Date Hdr - one of the 2 start date fields
        Rev REc Contract Start Date Item- one of the 2 start date fields
        Completed Sales (DC) - the document currency billings amount
        Completed Sales - the USD equivalent of the document currency billing

    There are two fields that may contain the start date of the contract and 2 dates that may contain the end date
    of the contract. (No idea why.) We need to take the columns for these fields that contains dates and
    then determine the number of months between the start and end date of each contract. This amount is then
    mapped into {"1M", "3M", "6M", "1Y", "2Y", "3Y"} based on which billing frequency is closest to the actual
    number of months between the contract start and end date.

    Once the contract date is determined, we can not assume that the contract will renew at the end of the contract term
    (this is primarilly becauase we have net new bookings from FP&A)

    :param config_dict: This is the main configuration dictionary for the deferred revenue program

    :param df: This is the billings dataframe for billings that have already been classified. It contains the same
                fields as the df dataframe that gets returned.

    :param model_dict: This dictionary contains curreneies that have been banned from the program to make sure that
                        they remain banned from the type_A_no_config billings we are processing here.
    :return: df: A dataframe containing all billings that have been classified before and during this function.
                The dataframe contains the following columns
                'curr': The three digit currency for the row
                'BU': The Enterprise BU for the billing
                'period': The period of the billing in YYYY-PP format
                'recognized_DC': The recognized document currency billings that go immediately to revenue
                'recognized_US': The USD equivalent of these immediate revenue billings
                'service_DC': The document currency billings for service billings that get placed into deferred and are
                            moved to revenue as service is performed
                'service_US': The USD equivalent of the service billings
                'deferred_1M_DC': document currency deferred billings that will renew every month
                'deferred_1M_US': USD equivalent of deferred 1M billings
                'deferred_3M_DC': document currency deferred billings that will renew every 3 months
                'deferred_3M_US': USD equivalent of deferred 3M billings
                'deferred_6M_DC':document currency deferred billings that will renew every 6 months
                'deferred_6M_US': USD equivalent of deferred 6M billings
                'deferred_1Y_DC': document currency deferred billings that will renew every year
                'deferred_1Y_US': USD equivalent of deferred annual billings
                'deferred_2Y_DC': document currency deferred billings that will renew every 2 years
                'deferred_2Y_US': USD equivalent of deferred 2Y billings
                'deferred_3Y_DC': document currency deferred billings that will renew every 3 year
                'deferred_3Y_US': USD equivalent of deferred 3Y billings

    '''
    billings_filename = config_dict['path_to_data'] + config_dict['billings']['filename']
    type_A_sheetname = config_dict['billings']['type_A_sheetname']

    temp_flat_DC, temp_flat_US = load_and_clean_type_A(
        billings_filename, model_dict, type_A_sheetname
    )

    df_billings = merge_billings_with_A(temp_flat_DC, temp_flat_US, df)

    return df_billings


def load_and_clean_type_A(billings_filename, model_dict, type_A_sheetname="type_A_no_config"):
    '''

    :param billings_filename: the filename for the base_billings file
    :param model_dict: The main model dictionary that contains banned currencies
    :param type_A_sheetname: The sheetname that contains the type_A_no_config billings
    :return: temp_flat_DC: a dataframe containing the document currency billings mapped to their rebillings frequency
    :return: temp_flat_US: a dataframe containing the US equivalent billings mapped to their rebillings frequency
    '''
    df_A = pd.read_excel(billings_filename, sheet_name=type_A_sheetname)

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

    # Removing banned currencie
    df_A = remove_bad_currencies(df_A, model_dict)

    # ###### Handling the duplicate dates by taking a max and creating a start_date and end_date fields in pandas datetime format
    # The type A billings file contains two different fields for the start date of each contract and two different fields for the end date of the contract. I don't know what determines which date fields a contract is entered into, but both date fields are necessary (usually the other date field is blank on a contract.) Here we are handling the duplicate date fields
    df_A["start_date_str"] = (
        df_A[["start_date_1", "start_date_2"]].max(axis=1).astype(str)
    )
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

    # Creating a month_interval field that calculates the difference between the start_date and end_date in months. We will map this number of months into a rebilling frequency (this number of months determines when the contract expires and the deferred revenue model assumes that all attribution is accounted for in our net new billings estimates provided by FP&A)
    df_A["month_interval"] = df_A["end_date"] - df_A["start_date"]
    df_A["months"] = (df_A["month_interval"] / np.timedelta64(1, "M")).round(0)

    df_A.head(10)

    # Mapping the number of months into our common rebill frequencies (monthly, quarterly, semi-annual, annual, 2 years and 3 years)
    list_rebills = [1, 3, 6, 12, 24, 36]
    temp_rebill = np.zeros_like(df_A["months"])
    for i in range(len(df_A)):
        temp_rebill[i] = min(list_rebills, key=lambda x: abs(x - df_A["months"][i]))
    df_A["rebill_months"] = temp_rebill

    # Dropping the columns we no longer need
    df_A.drop(
        columns=["start_date", "end_date", "month_interval", "months"],
        axis=1,
        inplace=True,
    )

    # Grouping the dataframe by rebill_months using a pivot table
    temp_DC = df_A.pivot_table("DC_amount", ["curr", "BU", "period"], "rebill_months")
    temp_US = df_A.pivot_table("US_amount", ["curr", "BU", "period"], "rebill_months")

    # ###### Filling in any zeros that arise if there is no contract on a specific period, currency and BU for a particular rebill period
    temp_DC = temp_DC.fillna(0)
    temp_US = temp_DC.fillna(0)

    # ###### Flattening the pivot table back to a normal dataframe and renaming the columns
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

    temp_flat_DC.head(20)

    return temp_flat_DC, temp_flat_US


def merge_billings_with_A(temp_flat_DC, temp_flat_US, df):
    '''
    Merging the billings dataframe with the temp_flat_DC and temp_flat_US dataframes and filling in any blanks with zero

    :param temp_flat_DC: a dataframe containing the DC billings of type_A billings with no valid config type

    :param temp_flat_US: a dataframe containing the US billings of type_A billings with no valid config type

    :param df: the billings dataframe containing the same columns as the column returned

    :return: a dataframe with the merged dataframes containing the following columns
                'curr': The three digit currency for the row
                'BU': The Enterprise BU for the billing
                'period': The period of the billing in YYYY-PP format
                'recognized_DC': The recognized document currency billings that go immediately to revenue
                'recognized_US': The USD equivalent of these immediate revenue billings
                'service_DC': The document currency billings for service billings that get placed into deferred and are
                            moved to revenue as service is performed
                'service_US': The USD equivalent of the service billings
                'deferred_1M_DC': document currency deferred billings that will renew every month
                'deferred_1M_US': USD equivalent of deferred 1M billings
                'deferred_3M_DC': document currency deferred billings that will renew every 3 months
                'deferred_3M_US': USD equivalent of deferred 3M billings
                'deferred_6M_DC':document currency deferred billings that will renew every 6 months
                'deferred_6M_US': USD equivalent of deferred 6M billings
                'deferred_1Y_DC': document currency deferred billings that will renew every year
                'deferred_1Y_US': USD equivalent of deferred annual billings
                'deferred_2Y_DC': document currency deferred billings that will renew every 2 years
                'deferred_2Y_US': USD equivalent of deferred 2Y billings
                'deferred_3Y_DC': document currency deferred billings that will renew every 3 year
                'deferred_3Y_US': USD equivalent of deferred 3Y billings


    '''
    # ###### Merging the billings dataframe with the temp_flat_DC dataframe and and temp_flat_US dataframe and filling in any blanks with zero
    df_with_A = pd.merge(
        df,
        temp_flat_DC,
        how="outer",
        left_on=["curr", "BU", "period"],
        right_on=["curr", "BU", "period"],
        indicator=True,
        validate="one_to_one",
    )

    df_with_A = df_with_A.fillna(
        pd.Series(0, index=df_with_A.select_dtypes(exclude="category").columns)
    )

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

    # ###### Combining columns form the different data sources (they get merged with different names) and cleaning up the columns
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

    # ###### Checking totals to se if they match what we expect
    print("sum of temp flat DC 1M:      ", temp_flat_DC["deferred_1M_DC"].sum())
    print("sum of base_df before DC 1M: ", df["deferred_1M_DC"].sum())
    print("sum of final DC 1M:          ", df_with_all["deferred_1M_DC"].sum())

    a = temp_flat_DC["deferred_1M_DC"].sum()
    b = df["deferred_1M_DC"].sum()
    c = df_with_all["deferred_1M_DC"].sum()
    print(c)
    print(a + b)

    # # TO BE DONE: Create a table that contains the total billings by DC for each dataframe and each step for auditing
    #
    #  - start with all of the DC
    #  - then create function that appends and adds rows
    #  - then do the same for the DC stuff type_A
    #  - then check the totals
    #

    # ##### Renaming the cleaned billings dataframe as df_billings
    df_billings = df_with_all.copy()

    df_billings = df_billings.sort_values(
        ["curr", "BU", "period"], ascending=(True, True, True)
    )

    return df_billings

def interp_FX_fwds(df_FX_rates):
    """ Creates monthly interpolated rates from the df_FX_rates file and adds the is_direct field """
    # Create list of tickers to determine which is direct (if USD is the first currency, it is direct)
    tickers = df_FX_rates["Ticker"].copy()
    first_curr = [sub[:-3] for sub in tickers]
    is_direct = []
    for curr in first_curr:
        if curr == "USD":
            is_direct.append(0)
        else:
            is_direct.append(1)

    df_FX_rates["is_direct"] = is_direct

    # Add new columns that will hold the forward rates
    new_cols = [
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

    for item in new_cols:
        df_FX_rates[item] = 0

    # Interpolate the forward rates
    interp_time = np.arange(1, 13)
    interp_time = interp_time / 12

    fwd_times = [0, 0.25, 0.5, 0.75, 1]

    for index, row in df_FX_rates.iterrows():
        fwds = [row["Spot"], row["FWD_3M"], row["FWD_6M"], row["FWD_9M"], row["FWD_1Y"]]
        interp_fwds = np.interp(interp_time, fwd_times, fwds)
        for i in np.arange(len(new_cols)):
            df_FX_rates.loc[index, new_cols[i]] = interp_fwds[i]

    return df_FX_rates


def test_df_duplicates(df):
    '''
    Testing whether we have duplicates in our merged dataframe
    '''
    df_test_dup = df.copy()
    orig_len = len(df_test_dup)
    print("Original Length of the dataframe before duplicate test: ", orig_len)

    df_test_dup = df_test_dup.drop_duplicates(subset=['curr', 'BU', 'period'])
    print('New length of database after duplicates have been removed: ', len(df_test_dup))

    if orig_len != len(df_test_dup):
        print('We had duplicates in the dataframe! Look into why')
        error_duplicates = 1
    else:
        error_duplicates = 0
    return error_duplicates


def merge_bookings_with_curr(df_bookings, df_curr_map):
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

    return df_bookings


def build_booking_periods(df_bookings, df_billings):
    '''
    STEPS
        Creating data to add to the billings dataframe to incorporate period by period billings
        Fills in the df_book_period dataframe with the quarterly bookings numbers for each BU and currency
        Creating lists of periods and quarters needed to fill out the df_book_period dataframe
        Adding the booking periods to the dataframe. The bookings are split into periods based on last years percentage of 1 year deferred billings within the quarter.
            For example: P1 = 25%, P2 = 30%, P3 = 45% such that the sum is equal to the total quarterly billings last year
        Cleaning up the dataframe by dropping the columns we no longer need
    '''
    #BELOW WAS FOR WHEN KAREN DID THE BILLINGS AND WE HAD BU as Creative, Doc Cloud, etc
    #list_BUs = df_bookings["BU"].unique()
    # this is now called segment
    list_BUs = df_bookings["BU"].unique()
    list_curr = df_bookings["Currency"].unique()

    print("This is the list of BUs in the bookings dataframe: ", list_BUs)
    print("This is the list of currencies in the bookings dataframe: ", list_curr)

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

    #df_book_period.head(14)

    #df_bookings.BU.value_counts()

    # Fills in the df_book_period dataframe with the quarterly bookings numbers for each BU and currency
    # fill in the quarters
    for i in range(len(df_book_period["BU"])):
        this_BU = df_book_period["BU"][i]
        this_curr = df_book_period["curr"][i]
        this_slice = df_bookings[
            (df_bookings["BU"] == this_BU) & (df_bookings["Currency"] == this_curr)
            ]

        this_Q1 = this_slice[this_slice["Quarter"] == "Q1_2021"]
        sum_Q1 = this_Q1["value"].sum()
        df_book_period["Q1"].loc[i] = sum_Q1

        this_Q2 = this_slice[this_slice["Quarter"] == "Q2_2021"]
        sum_Q2 = this_Q2["value"].sum()
        df_book_period["Q2"].loc[i] = sum_Q2

        this_Q3 = this_slice[this_slice["Quarter"] == "Q3_2021"]
        sum_Q3 = this_Q3["value"].sum()
        df_book_period["Q3"].loc[i] = sum_Q3

        this_Q4 = this_slice[this_slice["Quarter"] == "Q4_2021"]
        sum_Q4 = this_Q4["value"].sum()
        df_book_period["Q4"].loc[i] = sum_Q4

    df_book_period.head(30)

    print("Q1 total bookings ", df_book_period["Q1"].sum())
    print("Q2 total bookings ", df_book_period["Q2"].sum())
    print("Q3 total bookings ", df_book_period["Q3"].sum())
    print("Q4 total bookings ", df_book_period["Q4"].sum())

    # ##### Creating lists of periods and quarters needed to fill out the df_book_period dataframe
    # list of quarters for the percentages

    #TODO: Add these to the config file!!!
    list_q1 = ["2021-01", "2021-02", "2021-03", "2021-04", "2021-05", "2021-06",
               "2021-07", "2021-08", "2021-09","2021-10", "2021-11", "2021-12",
               "2021-13", "2021-14"]
    list_q2 = ["2021-15", "2021-16", "2021-17", "2021-18", "2021-19", "2021-20",
               "2021-21", "2021-22", "2021-23","2021-24", "2021-25", "2021-26",
               "2021-27"]
    list_q3 = ["2021-28", "2021-29", "2021-30", "2021-31", "2021-32", "2021-33",
               "2021-34", "2021-35", "2021-36", "2021-37", "2021-38", "2021-39",
               "2021-40"]
    list_q4 = ["2021-41", "2021-42", "2021-43", "2021-44", "2021-45", "2021-46",
               "2021-47", "2021-48", "2021-49", "2021-50", "2021-51", "2021-52",
               "2021-53"]

    # list periods should be the year prior periods. This is what gets looked up in df_billiings
    list_periods = [
        "2020-01",
        "2020-02",
        "2020-03",
        "2020-04",
        "2020-05",
        "2020-06",
        "2020-07",
        "2020-08",
        "2020-09",
        "2020-10",
        "2020-11",
        "2020-12",
        "2020-13",
        "2020-14",
        "2020-15",
        "2020-16",
        "2020-17",
        "2020-18",
        "2020-19",
        "2020-20",
        "2020-21",
        "2020-22",
        "2020-23",
        "2020-24",
        "2020-25",
        "2020-26",
        "2020-27",
        "2020-28",
        "2020-29",
        "2020-30",
        "2020-31",
        "2020-32",
        "2020-33",
        "2020-34",
        "2020-35",
        "2020-36",
        "2020-37",
        "2020-38",
        "2020-39",
        "2020-40",
        "2020-41",
        "2020-42",
        "2020-43",
        "2020-44",
        "2020-45",
        "2020-46",
        "2020-47",
        "2020-48",
        "2020-49",
        "2020-50",
        "2020-51",
        "2020-52",

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
        "P13",
        "P14",
        "P15",
        "P16",
        "P17",
        "P18",
        "P19",
        "P20",
        "P21",
        "P22",
        "P23",
        "P24",
        "P25",
        "P26",
        "P27",
        "P28",
        "P29",
        "P30",
        "P31",
        "P32",
        "P33",
        "P34",
        "P35",
        "P36",
        "P37",
        "P38",
        "P39",
        "P40",
        "P41",
        "P42",
        "P43",
        "P44",
        "P45",
        "P46",
        "P47",
        "P48",
        "P49",
        "P50",
        "P51",
        "P52",
        "P53",

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

    df_book_period.tail(10)

    # ###### Cleaning up the dataframe by dropping the columns we no longer need
    df_book_period.drop(
        ["bill_Q1_sum", "bill_Q2_sum", "bill_Q3_sum", "bill_Q4_sum"], axis=1, inplace=True
    )

    return df_book_period


def convert_bookings_to_DC(df_book_period, df_FX_fwds):
    '''
    The bookings forecast is in USD, but we need document currency bookings.
    This function takes the df_book_period dataframe and adds the DC equivalent
    to all of the bookings by using the plan FX forward rates dataframe df_FX_fwds

    '''
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

    # The df_book_period dataframe now has columns for bookings each period
    # in both local currency and document currency

    return df_book_period

def split_hybrid_dataframe(df_hyb, config_dict):
    '''
    The orders with a 'Sales Document Type' of "BNDL" are a combination of both the recognized revenue
    and deferred revenue. The historical percentage of "BNDL" orders thar are deferred is about 87%.

    This function takes the "BNDL" sales types (in a dataframe called df_hyb for hybrid) and
    splits this dataframe into two separate dataframes, one that is immediate revenue and one that is
    deferred.

    :param df_hyb: This is the billings dataframe for the hybrid ("BNDL") sales document type orders
    :param config_dict: This is the json config file. There is a field in this file called 'hybrid_pct_IR' which
            contains the percentage of the BNDL billings that will go straight to revenue.
    :return:
        df_IR - the dataframe with just the immediate revenue portion of the billings
        df_deferred - the dataframe that contains the deferred portion of the dataframe
    '''
    pct_IR = config_dict['hybrid_pct_IR']
    df_IR = df_hyb.copy()
    df_deferred = df_hyb.copy()

    df_IR['DC_amount'] = df_IR['DC_amount'] * pct_IR
    df_IR['US_amount'] = df_IR['US_amount'] * pct_IR

    df_deferred['DC_amount'] = df_deferred['DC_amount'] * (1 - pct_IR)
    df_deferred['US_amount'] = df_deferred['US_amount'] * (1 - pct_IR)

    return df_IR, df_deferred

def process_type_A(config_dict, df):
    '''
    Type A Billings

    These billings are on a billing plan. The product config tells us how long before they renew
    and the sub term determines how often they are billed. (See the document "Deferred Revenue Model August 2020"
    in the docs subfolder for a better understanding of the details.

    1M: config = 'MTHLY' or sub_term = 1
    1Y: config = '1Y' AND sub term = 0 or 12  OR any config with sub_term = 12
    2Y: config = '2Y' AND sub term = 0 or 24
    3Y: config = '3Y' AND sub term = 0 or 36
    There are also config types that do not allow us to map these into a billings frequency.
    These types are {"BLANK", "OCONS", "ONORE", "OUNIV"}
        These types are loaded from the type_A_no_config report, but we pass them back to the main
        program to test that we have not dropped anything.

    :param config_dict: This is the json config file. It contains a list 'type_A_config_keepers' that
            has the list of config types we know how to process (i.e. we know how they will be rebilled)

    :param df: This is the dataframe of deferred billings that have revenue recognition type A
    :return: return_A_dict: A dictionary containing dataframes containing the billings that have the same
                rebill frequency.
                return_A_dict = {'gb_a_1M': gb_a_1M,
                     'gb_a_1Y': gb_a_1Y,
                     'gb_a_2Y': gb_a_2Y,
                     'gb_a_3Y': gb_a_3Y,
                     'gb_a_no_config': gb_a_no_config,}
    '''

    # NOTE: groupby will completely ignore any blank terms and they will NOT appear in the
    # final groupby object. If there is a config or sub_term that is blank, we need to replace
    # this subterm with 'Blank' to make sure it works.
    df['config'].fillna('Blank', inplace=True)
    print('A config value counts')
    print(df['config'].value_counts())
    # grouping by fields we need to keep.
    gb_a = df.groupby(["curr", "BU", "period", "config", "sub_term"], as_index=False).sum()

    print('A config grouped value counts')
    print(gb_a["config"].value_counts(dropna=False))

    # splitting into config types we keep and ones we need to get in the type_A_no_config report
    config_type_keepers = config_dict['type_A_config_keepers']
    gb_a_keepers = gb_a[gb_a["config"].isin(config_type_keepers)].copy()
    gb_a_no_config = gb_a[~gb_a["config"].isin(config_type_keepers)].copy()

    print('len gb_a', len(gb_a))
    print('gb_a_keepers', len(gb_a_keepers))
    print('len df_a_no_config', len(gb_a_no_config))
    print('Total USD Equivalent Billings of Type A with bad configs', gb_a_no_config.US_amount.sum())

    # ###### Grouping by the config type into gb_a_1Y, gb_a_2Y, gb_a_3y, gb_a_1M dataframes
    # Selecting monthly billings
    gb_a_1M = gb_a_keepers[(gb_a_keepers['config'] == 'MTHLY') |
                           (gb_a_keepers['sub_term'] == 1)].copy()
    index_1M = gb_a_1M.index
    # dropping monthly billings from the keepers
    gb_a_keepers.drop(index_1M, inplace=True)

    gb_a_1Y = gb_a_keepers[(gb_a_keepers['sub_term'] == 12) |
                           ((gb_a_keepers['sub_term'] == 0) &
                            (gb_a_keepers['config'] == '1Y'))].copy()
    index_1Y = gb_a_1Y.index
    gb_a_keepers.drop(index_1Y, inplace=True)

    # remaining types will be 2Y with sub_term = 0 or 24 or 3Y with 0 or 36
    gb_a_2Y = gb_a_keepers[gb_a_keepers['config'] == '2Y'].copy()
    gb_a_3Y = gb_a_keepers[gb_a_keepers['config'] == '3Y'].copy()

    print("this is the length of type A 1M billings: ", len(gb_a_1M))
    print("this is the length of type A 1Y billings: ", len(gb_a_1Y))
    print("this is the length of type A 2Y billings: ", len(gb_a_2Y))
    print("this is the length of type A 3Y billings: ", len(gb_a_3Y))

    # Cleaning up the sub_term columns from the gb_A_#X variabels
    gb_a_1M.drop(labels=['sub_term'], axis=1, inplace=True)
    gb_a_1Y.drop(labels=['sub_term'], axis=1, inplace=True)
    gb_a_2Y.drop(labels=['sub_term'], axis=1, inplace=True)
    gb_a_3Y.drop(labels=['sub_term'], axis=1, inplace=True)

    # need to group these so we do not have duplicate BU, curr, period pairs
    gb_a_1M = gb_a_1M.groupby(['curr', 'BU', 'period'], as_index=False).sum()
    gb_a_1Y = gb_a_1Y.groupby(['curr', 'BU', 'period'], as_index=False).sum()
    gb_a_2Y = gb_a_2Y.groupby(['curr', 'BU', 'period'], as_index=False).sum()
    gb_a_3Y = gb_a_3Y.groupby(['curr', 'BU', 'period'], as_index=False).sum()

    return_A_dict = {'gb_a_1M': gb_a_1M,
                     'gb_a_1Y': gb_a_1Y,
                     'gb_a_2Y': gb_a_2Y,
                     'gb_a_3Y': gb_a_3Y,
                     'gb_a_no_config': gb_a_no_config,}
    return return_A_dict

def clean_curr_and_zeros(df, config_dict):
    # Removing currencies with less than n entries or that we have no information on
    banned_curr = config_dict['curr_banned']
    vc = df["curr"].value_counts()
    keep_these = vc.values > 20
    keep_curr = vc[keep_these]
    list_keepers = keep_curr.index
    remove_these = vc[vc.values <= 20].index

    #Adding banned currencies to the remove_these list
    for curr in banned_curr:
        if curr not in remove_these:
            remove_these.append(curr)

    model_dict = {"curr_removed": list(vc[remove_these].index)}

    #remove the banned currencies from the keepers list
    for curr in banned_curr:
        if curr in list_keepers:
            list_keepers.pop(curr)
    print('List of currencies kept')
    print(list_keepers)

    df = df[df["curr"].isin(list_keepers)]

    # clearing out zero amounts
    df = df[df["DC_amount"] != 0]

    return df, model_dict

def process_type_D(config_dict, df):
    '''
    TYPE D billings
    These billings have a field 'Rule For Bill Date' that determines when new billings will occur
     - Monthly:        *{Y1, Y2, Y3, Y5}*
     - Quarterly:      *YQ*
     - Every 4 months: *YT*  --NOTE: There are only 10 of these, so I am treating these as quarterly--
     - Semi-annual:    *YH*
     - Annual:         *{YA, YC}*
     - Every 2 years:  *Y4*
     - Every 3 years:  *Y7*

     We also need to track the type D billings that do not have a 'Rule for Bill Date'

    :param config_dict: This is the json config file. It contains a dictionary 'type_D_config_classification'
        that has the list of rebill_rules we know how to process (i.e. we know when the contract will end)

    :param df: This is the dataframe of deferred billings that have revenue recognition type D

    :return: return_D_dict: A dictionary containing dataframes containing the billings that have the same
                rebill_rule frequency.
                return_dict = {'monthly': gb_d_mthly,
                    'qtrly': gb_d_qtrly,
                   'semi_ann': gb_d_semi_ann,
                   'annual': gb_d_annual,
                   'two_years': gb_d_two_yrs,
                   'three_years': gb_d_three_yrs,
                   'no_rebill': gb_d_no_rebill}

    '''
    #  We also need to track the type D billings that do not have a 'Rule for Bill Date'
    gb_d = df.groupby(["curr", "BU", "period", "rebill_rule", "sales_doc"], as_index=False).sum()
    gb_d.drop(labels=["sub_term"], axis=1, inplace=True)

    gb_d["rebill_rule"].value_counts(dropna=False)

    # ###### Grouping these by rebill rule and incorporating rebill rules that have the same rebill period
    list_monthly = config_dict['type_D_classification']["list_monthly"]
    list_qtrly = config_dict['type_D_classification']["list_qtrly"]
    list_semi_ann = config_dict['type_D_classification']["list_semi_ann"]
    list_ann = config_dict['type_D_classification']["list_ann"]
    list_2yrs = config_dict['type_D_classification']["list_2yrs"]
    list_3yrs = config_dict['type_D_classification']["list_3yrs"]
    list_all_rebills = list_monthly + list_qtrly + list_semi_ann + list_ann + list_2yrs + list_3yrs

    gb_d_mthly = gb_d[gb_d["rebill_rule"].isin(list_monthly)].copy()
    gb_d_mthly.drop(labels="rebill_rule", axis=1, inplace=True)
    gb_d_mthly = gb_d_mthly.groupby(["curr", "BU", "period"]).sum()
    gb_d_mthly.reset_index(inplace=True)

    gb_d_qtrly = gb_d[gb_d["rebill_rule"].isin(list_qtrly)].copy()
    gb_d_qtrly.drop(labels="rebill_rule", axis=1, inplace=True)
    gb_d_qtrly = gb_d_qtrly.groupby(["curr", "BU", "period"]).sum()
    gb_d_qtrly.reset_index(inplace=True)

    gb_d_semi_ann = gb_d[gb_d["rebill_rule"].isin(list_semi_ann)]
    gb_d_semi_ann.drop(labels="rebill_rule", axis=1, inplace=True)
    gb_d_semi_ann = gb_d_semi_ann.groupby(["curr", "BU", "period"]).sum()
    gb_d_semi_ann.reset_index(inplace=True)

    gb_d_annual = gb_d[gb_d["rebill_rule"].isin(list_ann)].copy()
    gb_d_annual.drop(labels="rebill_rule", axis=1, inplace=True)
    gb_d_annual = gb_d_annual.groupby(["curr", "BU", "period"]).sum()
    gb_d_annual.reset_index(inplace=True)

    gb_d_two_yrs = gb_d[gb_d["rebill_rule"].isin(list_2yrs)].copy()
    gb_d_two_yrs.drop(labels="rebill_rule", axis=1, inplace=True)
    gb_d_two_yrs = gb_d_two_yrs.groupby(["curr", "BU", "period"]).sum()
    gb_d_two_yrs.reset_index(inplace=True)

    gb_d_three_yrs = gb_d[gb_d["rebill_rule"].isin(list_3yrs)]
    gb_d_three_yrs.drop(labels="rebill_rule", axis=1, inplace=True)
    gb_d_three_yrs = gb_d_three_yrs.groupby(["curr", "BU", "period"]).sum()
    gb_d_three_yrs.reset_index(inplace=True)

    gb_d_no_rebill = gb_d[~gb_d['rebill_rule'].isin(list_all_rebills)].copy()

    print('There are {} line items that are type D and have no rebill rule'.format(len(gb_d_no_rebill)))
    print("Length of monthly", len(gb_d_mthly))
    print("Length of quarterly", len(gb_d_qtrly))
    print("Length of semi ann", len(gb_d_semi_ann))
    print("Length of annual", len(gb_d_annual))
    print("Length of two years", len(gb_d_two_yrs))
    print("Length of three years", len(gb_d_three_yrs))

    return_dict = {'monthly': gb_d_mthly,
                   'qtrly': gb_d_qtrly,
                   'semi_ann': gb_d_semi_ann,
                   'annual': gb_d_annual,
                   'two_years': gb_d_two_yrs,
                   'three_years': gb_d_three_yrs,
                   'no_rebill': gb_d_no_rebill}
    return return_dict

def load_base_billings(config_dict):
    """
    This loads up the base billings data and creates the following:
    1. the main billing dataframe (df)
    2. a dataframe of billings that need to be further classified because they are missing a POB_type field (df_no_POB)
    3. a dataframe that contains type A deferred billings that cannot be classified (df_type_a_no_config)
    4.  a dataframe that contains any deferred type D billings that have no rebill rule and need to be reclassified.

    :param config_dict: This is the json config file.
    :return:
        df: This is the main dataframe for the billings we have classified. The dataframe contains the following columns
                'curr': The three digit currency for the row
                'BU': The Enterprise BU for the billing
                'period': The period of the billing in YYYY-PP format
                'recognized_DC': The recognized document currency billings that go immediately to revenue
                'recognized_US': The USD equivalent of these immediate revenue billings
                'service_DC': The document currency billings for service billings that get placed into deferred and are
                            moved to revenue as service is performed
                'service_US': The USD equivalent of the service billings
                'deferred_1M_DC': document currency deferred billings that will renew every month
                'deferred_1M_US': USD equivalent of deferred 1M billings
                'deferred_3M_DC': document currency deferred billings that will renew every 3 months
                'deferred_3M_US': USD equivalent of deferred 3M billings
                'deferred_6M_DC':document currency deferred billings that will renew every 6 months
                'deferred_6M_US': USD equivalent of deferred 6M billings
                'deferred_1Y_DC': document currency deferred billings that will renew every year
                'deferred_1Y_US': USD equivalent of deferred annual billings
                'deferred_2Y_DC': document currency deferred billings that will renew every 2 years
                'deferred_2Y_US': USD equivalent of deferred 2Y billings
                'deferred_3Y_DC': document currency deferred billings that will renew every 3 year
                'deferred_3Y_US': USD equivalent of deferred 3Y billings

        model_dict: This dictionary contains the list of currencies that we exclude from the program

        df_no_POB: This dataframe contains the billings data that had no POB type classification and were not able to be
                classified in this function (because this field was missing). The format of this file is different than
                the df dataframe because the data is still in the excel format from the 'all_billings_inputs.xlsx' file
                that gets pulled from tableau (actually the 'base_billings' tab in that workbook).
                The df_no_POB dataframe contains the following fields:
                    'curr': The 3character currency ticker for this row of billings
                    'BU': The enterprise document currency
                    'period': The periods of the billing in YYYY-PP format
                    'POB_type': The POB_type classification (in the df_no_POB file, these are all blank)
                    'config': This is the type A configuration {'MTHLY', '1Y', '2Y', '3Y', 'OUCONS', 'ONORE', BLANK}
                    'rev_req_type': This is the revenue recognition classification (pre 606) {'A', 'B', 'D', 'F', BLANK}
                    'rebill_rule': This is the field that classifies the rev_req_type == D billings. The list of possible
                                rebill_rules is contained in the base_config.json file
                    'sales_doc': For some of these billings that are missing important fields, we use the sales_doc type
                                to classify their type (deferred, service or immediate) and rebillings frequency
                    'sales_type': This is an old field that was used before POB_type became implemented with 606.
                                Pre 606, this was used to classify billings as being deferred, service or immediate rev.
                                The field is kept here to attempt to classify billings that are missing data fields
                    'sub_term': The sub term is the length of the subscription (mostly for type D billings). This field
                                is used in conjunction with the rebill_rule to determine billings frequency
                    'DC_amount': The document currency amount of the billing
                    'US_amount': The USD eqiuvalent of the document currency billing

        gb_a_no_config: This dataframe contains deferred billings (so they have a POB_type) that have a
                        revenue recognition type of A, but a product config that cannot be classified into a
                        rebilling frequency. The type A product configs {"MTHLY", '1Y', '2Y', '3Y'} can be classified.
                        The other product configs, including blanks, cannot be classified. The value of these billings
                        is then taken from the 'type_A_no_config' tab in the 'all_billings_inputs.xlsx' file.
                        The columns in this dataframe are the same as the df_no_POB dataframe.

        gb_d_no_rebill: This dataframe contains the billings that have a POB type that is deferred, have a revenue
                        recognition type of D and have no rebill_rule field. NOTE: This dataframe is usually empty, but
                        is being returned here to check.
    """
    filename_billings = config_dict['path_to_data'] + config_dict['billings']['filename']
    sheetname_billings = config_dict['billings']['base_sheetname']
    df = pd.read_excel(filename_billings, sheet_name=sheetname_billings)

    # Changing column names
    df.rename(
        index=str,
        columns={
            "Contrct Duration in Months": "duration",
            "Document Currency": "curr",
            "Enterprise BU Desc": "BU",
            #"Invoice Fiscal Year Period Desc": "period",
            "Invoice Calendar Year Week Desc": "period",
            "POB Type": "POB_type",
            "Product Config Type": "config",
            "Rev Rec Category": "rev_req_type",
            "Rule For Bill Date": "rebill_rule",
            "Sales Document Type": "sales_doc",
            "Sales Type": "sales_type",
            "Subscription Term": "sub_term",
            "Completed Sales ( DC )": "DC_amount",
            "Completed Sales": "US_amount",
        },
        inplace=True,
    )
    # getting rid of duration here (this field is used to classify service billings)
    df.drop(labels=["duration"], axis=1, inplace=True)
    df, model_dict = clean_curr_and_zeros(df, config_dict)

    # POB Type Classification
    list_IR = config_dict['POB_type_classifier']['list_IR']
    list_service = config_dict['POB_type_classifier']['list_service']
    list_deferred = config_dict['POB_type_classifier']['list_deferred']
    list_hybrid = config_dict['POB_type_classifier']['list_hybrid']
    list_all = list_IR + list_service + list_deferred + list_hybrid

    rec = df[df["POB_type"].isin(list_IR)].copy()
    svc = df[df["POB_type"].isin(list_service)].copy()
    dfr = df[df["POB_type"].isin(list_deferred)].copy()
    df_hyb = df[df["POB_type"].isin(list_hybrid)].copy()
    df_no_POB = df[~df["POB_type"].isin(list_all)].copy()

    # POB_type 'BNDL' is a mix of recognized revenue and deferred revenue. The function below
    # splits these billings into these two parts.
    df_hyb_IR, df_hyb_dfr = split_hybrid_dataframe(df_hyb, config_dict)

    # concatenate df_hyb_IR with rec and df_hyb_drf with df_hyb_drf
    rec = pd.concat([rec, df_hyb_IR])
    dfr = pd.concat([dfr, df_hyb_dfr])

    # Recognized Revenue
    # Below we are grouping the rec dataframe by Currency, Business Unit and Period and cleaning up the data we do
    # not need. Since the recognized revenue go directly to revenue, there is no contract that will renew and need
    # to be modeled in the future.

    # Grouping by currency, BU and period to save space
    gb_rec = rec.groupby(["curr", "BU", "period"], as_index=False).sum()
    gb_rec.drop(labels=["sub_term"] , axis=1, inplace=True)

    # Service Billings
    # Below we are grouping the svc dataframe by Currency, Business Unit and Period and
    # cleaning up the data we do not need. Since the service billings go directly to revenue,
    # there is no contract that will renew and need to be modeled in the future.
    # TODO: Take care of the service billings (how should they amortize?)
    # possibly only a problem when we get to deferred revenue (forecasting)
    gb_svc = svc.groupby(["curr", "BU", "period"], as_index=False).sum()
    gb_svc.drop(labels=["sub_term"], axis=1, inplace=True)

    # Deferred Billings
    # Splitting the deferred billings based on their revenue recognition type and processing the billings.
    #
    # Type A Deferred
    dfr_a = dfr[dfr["rev_req_type"] == "A"].copy()

    total_USD = dfr_a['US_amount'].sum()
    print("Total type A in USD: ", total_USD)

    A_df_dict = process_type_A(config_dict, dfr_a)

    gb_a_1M = A_df_dict['gb_a_1M']
    gb_a_1Y = A_df_dict['gb_a_1Y']
    gb_a_2Y = A_df_dict['gb_a_2Y']
    gb_a_3Y = A_df_dict['gb_a_3Y']
    gb_a_no_config = A_df_dict['gb_a_no_config']

    #
    # Type D Billings
    dfr_d = dfr[dfr["rev_req_type"] == "D"].copy()

    return_dict = process_type_D(config_dict, dfr_d)

    gb_d_mthly = return_dict['monthly']
    gb_d_qtrly = return_dict['qtrly']
    gb_d_semi_ann = return_dict['semi_ann']
    gb_d_annual = return_dict['annual']
    gb_d_two_yrs = return_dict['two_years']
    gb_d_three_yrs = return_dict['three_years']
    gb_d_no_rebill = return_dict['no_rebill']

    # Building a single dataframe that incorporates all of this data


    list_df = [
        gb_rec,
        gb_svc,
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

    return df, model_dict, df_no_POB, gb_a_no_config, gb_d_no_rebill

def classify_no_POB(config_dict, df):
    '''
    Many of the rows of data from the base billings file do not contain a POB_type classification.
    This function attempts to classify these billings both in revenue type {'Immediate', 'service', 'deferred'}
    and in the frequency of rebillings {'1M', '3M', '6M', '1Y', '2Y', '3Y'} basd on sales document type.

    The list of sales document types and their classifications is included in the base_config.json file
    and also explained in the 'Deferred Revenue Model August 2020.doc' file contained in the docs folder.
    Please review this before debugging this function.

    NOTE: If a billings has no POB_type AND no rev_req_type, then it is classified as immediate revenue.

    :param config_dict: The main dictionary for the deferred revenue model
    :param df: a dataframe containing the deferred billings that did not have a POB_type in the tableua
            extract contined in the 'all_base_billings.xlsx' file on the 'base_billlings' tab.
            The df_no_POB dataframe contains the following fields:
                    'curr': The 3character currency ticker for this row of billings
                    'BU': The enterprise document currency
                    'period': The periods of the billing in YYYY-PP format
                    'POB_type': The POB_type classification (in the df_no_POB file, these are all blank)
                    'config': This is the type A configuration {'MTHLY', '1Y', '2Y', '3Y', 'OUCONS', 'ONORE', BLANK}
                    'rev_req_type': This is the revenue recognition classification (pre 606) {'A', 'B', 'D', 'F', BLANK}
                    'rebill_rule': This is the field that classifies the rev_req_type == D billings. The list of possible
                                rebill_rules is contained in the base_config.json file
                    'sales_doc': For some of these billings that are missing important fields, we use the sales_doc type
                                to classify their type (deferred, service or immediate) and rebillings frequency
                    'sales_type': This is an old field that was used before POB_type became implemented with 606.
                                Pre 606, this was used to classify billings as being deferred, service or immediate rev.
                                The field is kept here to attempt to classify billings that are missing data fields
                    'sub_term': The sub term is the length of the subscription (mostly for type D billings). This field
                                is used in conjunction with the rebill_rule to determine billings frequency
                    'DC_amount': The document currency amount of the billing
                    'US_amount': The USD eqiuvalent of the document currency billing

    :return:df: a dataframe containing the classified billings that do not have a POB_type.
                The fields to this dataframe:
                'curr': The three digit currency for the row
                'BU': The Enterprise BU for the billing
                'period': The period of the billing in YYYY-PP format
                'recognized_DC': The recognized document currency billings that go immediately to revenue
                'recognized_US': The USD equivalent of these immediate revenue billings
                'service_DC': The document currency billings for service billings that get placed into deferred and are
                            moved to revenue as service is performed
                'service_US': The USD equivalent of the service billings
                'deferred_1M_DC': document currency deferred billings that will renew every month
                'deferred_1M_US': USD equivalent of deferred 1M billings
                'deferred_3M_DC': document currency deferred billings that will renew every 3 months
                'deferred_3M_US': USD equivalent of deferred 3M billings
                'deferred_6M_DC':document currency deferred billings that will renew every 6 months
                'deferred_6M_US': USD equivalent of deferred 6M billings
                'deferred_1Y_DC': document currency deferred billings that will renew every year
                'deferred_1Y_US': USD equivalent of deferred annual billings
                'deferred_2Y_DC': document currency deferred billings that will renew every 2 years
                'deferred_2Y_US': USD equivalent of deferred 2Y billings
                'deferred_3Y_DC': document currency deferred billings that will renew every 3 year
                'deferred_3Y_US': USD equivalent of deferred 3Y billings

            gb_a_no_config: contains rows that have no POB_type, are revenue recognition type A but have a
                            product_config type that cannot be classified. i.e. they are not {"MTHLY", '1Y", "2Y", "3Y"}
                            The fields to this dataframe are the same as the input dataframe df

            gb_d_no_rebill: contains rows that have no POB_type, are classified as revenue recognition type D and have
                            no rebill_rule field. NOTE: We expect this dataframe to be blank unless there is an issue
                            with the tableua data extracted to the base bilings file.
    '''
    # When there is no POB type, we need to classify based on sales document type
    # the dataframe df that gets passed in here is df_no_POB
    sales_doc_type = config_dict["sales_doc_type"]

    # Step 1: Split these based on the sales_doc_type
    rec_no_POB = df[df['sales_doc'].isin(sales_doc_type['immediate_revenue'])].copy()
    svc_no_POB = df[df['sales_doc'].isin(sales_doc_type['service'])].copy()
    dfr_no_POB = df[df['sales_doc'].isin(sales_doc_type['deferred'])].copy()

    # Try to classify the deferred billings based on rev req type
    # First I need to deal with the ZCSB sales type which are all type D
    # create slice/copy of dfr where rev_req_type == D OR sales_doc == ZCSB
    # then delete these from dfr
    dfr_d = dfr_no_POB[(dfr_no_POB["rev_req_type"] == "D") |
                        (dfr_no_POB['sales_doc']=='ZCSB')].copy()
    index_d_delete = dfr_d.index
    dfr_no_POB.drop(index_d_delete, inplace=True)

    # now classify the remainder based on if they are A or B or neither
    dfr_a = dfr_no_POB[dfr_no_POB["rev_req_type"] == "A"].copy()
    dfr_svc = dfr_no_POB[dfr_no_POB['rev_req_type']=='B'].copy()
    no_rev_req_type = dfr_no_POB[~dfr_no_POB['rev_req_type'].isin(['A', 'B', 'D'])].copy()

    print("These have no rev req type and will be treated as immediate revenue")
    print(no_rev_req_type)

    # combining rec_no_POB with no_rev_req_type (no_rev_req_type assume immediate revenue)
    df_rec = pd.concat([rec_no_POB, no_rev_req_type])
    gb_rec = df_rec.groupby(["curr", "BU", "period"]).sum()
    gb_rec.drop(labels=["sub_term"], axis=1, inplace=True)

    # combining the two types of service revenue
    # svc_no_POB is classified by sales_doc_type as being service
    # drf_svc is a deferred billings by sales_doc_type, but has rev_req_type == B, which is service
    df_svc = pd.concat([svc_no_POB, dfr_svc])
    gb_svc = df_svc.groupby(["curr", "BU", "period"]).sum()
    gb_svc.drop(labels=["sub_term"], axis=1, inplace=True)

    # processing the type A without a POB type
    df_dict_A = process_type_A(config_dict, dfr_a)
    gb_a_1M = df_dict_A['gb_a_1M']
    gb_a_1Y = df_dict_A['gb_a_1Y']
    gb_a_2Y = df_dict_A['gb_a_2Y']
    gb_a_3Y = df_dict_A['gb_a_3Y']
    gb_a_no_config = df_dict_A['gb_a_no_config']

    # processing the type B without a POB type
    return_dict = process_type_D(config_dict, dfr_d)
    gb_d_mthly = return_dict['monthly']
    gb_d_qtrly = return_dict['qtrly']
    gb_d_semi_ann = return_dict['semi_ann']
    gb_d_annual = return_dict['annual']
    gb_d_two_yrs = return_dict['two_years']
    gb_d_three_yrs = return_dict['three_years']
    gb_d_no_rebill = return_dict['no_rebill']

    list_df = [
        gb_rec,
        gb_svc,
        gb_a_1M,
        gb_a_1Y,
        gb_a_2Y,
        gb_a_3Y,
        gb_d_mthly,
        gb_d_qtrly,
        gb_d_semi_ann,
        gb_d_annual,
        gb_d_two_yrs,
        gb_d_three_yrs
    ]

    list_columns = [
        "recognized",
        "service",
        "deferred_1M_a",
        "deferred_1Y_a",
        "deferred_2Y_a",
        "deferred_3Y_a",
        "deferred_1M_d",
        "deferred_3M_d",
        "deferred_6M_d",
        "deferred_1Y_d",
        "deferred_2Y_d",
        "deferred_3Y_d"
    ]

    df = merge_all_dataframes(list_df, list_columns)

    df = clean_df_columns(df)

    return df, gb_a_no_config, gb_d_no_rebill

def load_and_clean_init_waterfall(config_dict):
    '''
    Loading up the deferred revenue waterfall from the revenue accounting workbook

    Loading the Deferred Revenue Forecast Sheet

    Steps to cleaning this notebook

    1.  Clear out the odd spacing in the Enterprise BU column
    2.  Remove the columns we do not need
    3.  Change the name of Marketo and Magento BU to Digital Experience
    4.  Aggregate this by External reporting BU
    5.  Rename the columns removing the odd punctuation
    6. Create interpolated periods here for the amortization (assume amortization to revenue is linear
        within the periods of a quarter)

    INPUTS:
        config_dict which contains the following
            "deferred_workbook": {
            "filename": "Q2'20 Rev Acctg Mgmt Workbook (06-04-20).xlsx",
            "sheetname": "Deferred Revenue Forecast",
            "skiprows": 5,
            "keeper_rows": [
                "Digital Media Total",
                  "Publishing Total",
                  "Digital Experience Total",
                  "Marketo Deferred",
                  "Magento Deferred",
                  "Grand Total inclusive of Magento/Marketo"
            ],
            "str_replace_from": ["Magento Deferred", "Marketo Deferred"],
            "str_replace_to": "Digital Experience Total",
            "drop_columns": [
                "Major Product Config", " Historical"
            ]
    OUTPUT:
        df_waterfall: This is a dataframe that contains the initial waterfall from the Accounting Workbook
            rows: Enterprice BUs
            columns: periods of amortization, as-performed/upon acceptance, and total deferred revenue
    '''
    path = config_dict["path_to_data"]
    filename = config_dict['deferred_workbook']['filename']
    sheetname = config_dict['deferred_workbook']['sheetname']
    skiprows = config_dict['deferred_workbook']['skiprows']
    path_and_filename= path + filename

    df = pd.read_excel(path_and_filename,
        sheet_name=sheetname,
        skiprows=skiprows)

    # some of the columns rows have blank spaces stripping this out
    df['External Reporting BU'] = df['External Reporting BU'].str.strip()

    # clearing out columns that we do not need
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    drop_columns = config_dict['deferred_workbook']['drop_columns']
    df = df.drop(columns=drop_columns)

    # clearing out rows we do not need
    keeper_rows = config_dict['deferred_workbook']['keeper_rows']
    df = df[df['External Reporting BU'].isin(keeper_rows)]

    '''
    Below was to account for Magento and Marketo deferred being listed separately. It is now in Digital Experience
    
    Below was in teh period_config under deferred_workbook:
     "str_replace_from": [
            "Magento Deferred",
            "Marketo Deferred"
        ],
        "str_replace_to": "Digital Experience Total",
    
    list_BU_changes = config_dict['deferred_workbook']["str_replace_from"]
    list_BU_new = config_dict['deferred_workbook']["str_replace_to"]
    for item in list_BU_changes:
        df["External Reporting BU"] = df["External Reporting BU"].str.replace(
            item, list_BU_new)
    '''

    '''
     
    Q2 2021: We now have Workfront, but it is included in the 'Grand Total incl. Workfront'
    We need to add the difference between the 'Grand Total' and the 'Grand Total incl. Workfront" and add to
    Digital Experience
    BELOW IS A ONE TIME CHANGE THAT I WILL HAVE TO MAKE INTO THE DICTIONARY AT SOME POINT
    
    df.loc['Digital Experience Total'] = df.loc['Digital Experience Total'] + df.loc['Grand Total incl Workfront'] -
        df.loc['Grant Total']
    
    
    '''
    df = df.replace('Workfront Total', 'Digital Experience Total')


    # Removing non-alphanumeric characters from column dates
    changed_columns = df.columns.str.replace("'", "_")
    changed_columns = changed_columns.str.replace("+", "")
    df.columns = changed_columns

    # Grouping by external reporting BU now that we have changed the BUs
    df = df.groupby("External Reporting BU").sum()

    # Adjust for the reporting in thousands (FP&A report)
    df = df * 1000

    # Creating the columns that have this amortization by period
    #
    # Note: My forecast looks at the end of period values always. The bookings forecast is quarterly,
    # which we change to be a monthly (level) bookings forecast. To arrive at the value of the
    # bookings forecast at the end of the first period, one period's worth (1/3) of a qarter,
    # has already been booked and need to be eliminated from the dataframe.
    #
    # To adjust for this, the bookings dataframe will have a P0 that contains the first month's
    # bookings and will be removed.
    #
    # This created an error the first time I tested the program that overstated the bookings,
    # billings and deferred.

    new_columns = []
    for i in range(12 * 3):
        if len(str(i)) == 1:
            new_column = "P0" + str(i)
        else:
            new_column = "P" + str(i)
        new_columns.append(new_column)

    qtrly_list = [col for col in df.columns if "Q" in col]

    period_index = 0
    for index, qtr in enumerate(qtrly_list):
        df[new_columns[period_index]] = df[qtr] / 3
        period_index += 1
        df[new_columns[period_index]] = df[qtr] / 3
        period_index += 1
        df[new_columns[period_index]] = df[qtr] / 3
        period_index += 1

    # OK My periods work fine. Now I can move on to saving this and finishing the defered waterfall

    df = df.loc[:, df.columns.str.contains("P")]

    # ##### Now dropping P0 from the waterfall
    df = df.drop("P00", axis=1)

    # Changing the BUs to match billings and dropping the total row
    df = df.rename(index=config_dict['waterfall_BU_mapping'])
    #df = df.drop(config_dict['waterfall_drop'])
    print(df)
    return df

def configure_df_fcst(df_billings, df_cal, config_dict):
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
