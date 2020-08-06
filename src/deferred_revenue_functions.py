"""
This file contains the functions used in the deferred revenue forecast
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.linear_model import LinearRegression

plt.style.use("ggplot")





def load_FX_data(FX_rates_filename, FX_rates_sheetname="to_matlab"):
    df_FX_rates = pd.read_excel(FX_rates_filename, sheet_name=FX_rates_sheetname)
    df_FX_rates["VOL_3M"] = df_FX_rates["VOL_3M"] / 100
    df_FX_rates["VOL_6M"] = df_FX_rates["VOL_6M"] / 100
    df_FX_rates["VOL_9M"] = df_FX_rates["VOL_9M"] / 100
    df_FX_rates["VOL_1Y"] = df_FX_rates["VOL_1Y"] / 100

    df_FX_rates.head(5)
    return df_FX_rates


def load_curr_map(curr_map_filename, curr_map_sheetname="curr_map"):
    df_curr_map = pd.read_excel(curr_map_filename, sheet_name=curr_map_sheetname)
    df_curr_map["Country"] = df_curr_map["Country"].str.replace(
        "\(MA\)", "", case=False
    )
    df_curr_map["Country"] = df_curr_map["Country"].str.strip()

    return df_curr_map


def add_billings_periods(df_billings):
    # clean up billings by removing LiveCycle and other solutions
    index_lc = df_billings[df_billings["BU"] == "LiveCycle"].index
    df_billings.drop(index_lc, inplace=True)

    index_other = df_billings[df_billings["BU"] == "Other Solutions"].index
    df_billings.drop(index_other, inplace=True)
    all_BU = df_billings["BU"].unique()
    all_curr = df_billings["curr"].unique()

    all_periods = df_billings["period"].unique()
    all_periods = np.sort(all_periods)
    all_periods = all_periods[-36:]

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


def load_ADBE_cal(ADBE_cal_filename, ADBE_cal_sheetname="ADBE_cal"):
    # loading Adobe financial calendar and calculating period weeks
    df_cal = pd.read_excel(ADBE_cal_filename, ADBE_cal_sheetname)
    df_cal["Period_Weeks"] = (df_cal["Per_End"] - df_cal["Per_Start"]) / np.timedelta64(
        1, "W"
    )
    df_cal["Period_Weeks"] = df_cal["Period_Weeks"].astype(int)
    df_cal["Period_Weeks"] = df_cal["Period_Weeks"] + 1

    # df_cal.head(5)
    # df_cal.sample(5)
    df_cal.tail(5)

    # Creating a column in df_cal with year  '-' the last two digits of the per_ticker to match with the billings dataframe
    df_cal["p2digit"] = df_cal["Period"].astype(str)
    df_cal["p2digit"] = df_cal["p2digit"].str.zfill(2)

    df_cal["period_match"] = (
            df_cal["Year"].astype(str) + "-" + df_cal["p2digit"].astype(str)
    )

    df_cal.drop(["p2digit"], axis=1, inplace=True)

    # df_cal.head(10)
    df_cal.sample(10)
    # df_cal.tail(10)

    df_cal.drop(
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

    return df_cal


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

        for col in list_columns:
            new_column = col + "US"
            old_column = col + "DC"

            DC_values = this_slice[old_column].values
            DC_values = DC_values.reshape(-1, 1)
            transp_fwds = transp_fwds.reshape(-1, 1)
            xx = DC_values * transp_fwds

            df_fcst.loc[
                (df_fcst["BU"] == this_BU) & (df_fcst["curr"] == this_curr), new_column
            ] = xx

    return df_fcst


def merge_bookings_to_fcst(df_book_period, df_fcst):
    dc_list = ["P07_DC", "P08_DC", "P09_DC", "P10_DC", "P11_DC", "P12_DC"]
    us_list = ["P07", "P08", "P09", "P10", "P11", "P12"]

    df_temp_book_period = df_book_period.drop(
        columns=[
            "Q1",
            "Q2",
            "Q3",
            "Q4",
            "P01",
            "P02",
            "P03",
            "P04",
            "P05",
            "P06",
            "P01_DC",
            "P02_DC",
            "P03_DC",
            "P04_DC",
            "P05_DC",
            "P06_DC",
            "FX_fwd_rate",
        ]
    ).copy()

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

    df_DC_melt["period"] = df_DC_melt["period"].str.replace("P", "2020-")
    df_DC_melt["period"] = df_DC_melt["period"].str.replace("_DC", "")
    df_US_melt["period"] = df_US_melt["period"].str.replace("P", "2020-")
    df_US_melt["period"] = df_US_melt["period"].str.replace("_US", "")

    # reset index
    df_DC_melt.set_index(["BU", "curr", "period"], inplace=True)

    df_US_melt.set_index(["BU", "curr", "period"], inplace=True)

    df_melted = df_DC_melt.join(df_US_melt, how="left")

    df_fcst.set_index(["BU", "curr", "period"], inplace=True)

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


def remove_bad_currencies(df, model_dict):
    this_list = df["curr"].unique().tolist()

    for curr in model_dict["curr_removed"]:
        if curr in this_list:
            print("Removed the following currency: ", curr)
            df = df[df["curr"] != curr]
    return df


def load_FX_fwds(FX_fwds_filename, FX_fwds_sheetname="forward_data"):
    df_FX_fwds = pd.read_excel(
        FX_fwds_filename, sheet_name=FX_fwds_sheetname, skiprows=1, usecols="C,G",
    )

    df_FX_fwds.rename(
        index=str, columns={"Unnamed: 2": "curr", "FWD REF": "forward"}, inplace=True
    )
    return df_FX_fwds


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
        this_type_B = df_slice.loc[
            df_slice["period"].isin(old_per_1Y), "deferred_B_DC"
        ].copy()
        df_fcst.loc[
            (df_fcst["BU"] == this_BU) & (df_fcst["curr"] == this_curr), "deferred_B_DC"
        ] = this_type_B.values

        # MONTHLY BILLINGS
        # here we need to call a seperate function using just the X array that is the one month billings
        this_y = df_slice["deferred_1M_DC"].copy()
        this_y = this_y.to_numpy()
        this_y = this_y.reshape(-1, 1)

        if sum(this_y) != 0:
            period_weeks = df_slice["Period_Weeks"].copy()
            period_weeks = period_weeks.to_numpy()
            period_weeks = period_weeks.reshape(-1, 1)

            this_y = np.true_divide(this_y, period_weeks)
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
            ] = this_model["intercept"]

            df_fcst.loc[
                (df_fcst["BU"] == this_BU) & (df_fcst["curr"] == this_curr), "coeff"
            ] = this_model["coeff"]

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

    # print("Model Score :",       best_score)
    # print("Model intercept :",   best_model.intercept_)
    # print("Model Coefficient :", best_model.coef_)

    for start_row in np.arange(1, y.shape[0] - 12):
        new_X = X[start_row:]
        new_y = y[start_row:]

        new_model = LinearRegression(fit_intercept=True)
        new_model.fit(new_X, new_y)
        new_score = new_model.score(new_X, new_y)
        new_int = new_model.intercept_
        new_coeff = new_model.coef_

        # print("Model Score :",       new_score)
        # print("Model intercept :",   new_model.intercept_)
        # print("Model Coefficient :", new_model.coef_)

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


def load_bookings(bookings_filename, bookings_sheetname="source"):
    df_bookings = pd.read_excel(bookings_filename, bookings_sheetname)

    df_bookings.head(10)
    # df_bookings.sample(10)
    # df_bookings.tail(10)

    # ### Cleaning up the bookings data
    # ##### NOTE: The bookings spreadsheet looks very different for Q2 versus prior quarters!
    #  - remove odd strings such as '(GP)' from BU, (IS) from Internal Segment, etc
    #  - dropping columns we do not need
    #  - renaming columns to better match our data naming convention
    #
    #  NOTE: The '('  and ')' is a special character so we need to precede these with the escape character '\'
    #
    #  NOTE: 2 The columns also have leading or trailing spaces, we need to strip them

    df_bookings["EBU"] = df_bookings["EBU"].str.replace(" \(GP\)", "", case=False)
    df_bookings["Internal Segment"] = df_bookings["Internal Segment"].str.replace(
        "\(IS\)", ""
    )
    df_bookings["PMBU"] = df_bookings["PMBU"].str.replace("\(PMBU\)", "")
    df_bookings["Geo"] = df_bookings["Geo"].str.replace("\(G\)", "")
    df_bookings["Market Area"] = df_bookings["Market Area"].str.replace("\(MA\)", "")
    df_bookings["Booking Type (Low)"] = df_bookings["Booking Type (Low)"].str.replace(
        "\(MA\)", ""
    )

    df_bookings["EBU"] = df_bookings["EBU"].str.strip()
    df_bookings["Internal Segment"] = df_bookings["Internal Segment"].str.strip()
    df_bookings["PMBU"] = df_bookings["PMBU"].str.strip()
    df_bookings["Geo"] = df_bookings["Geo"].str.strip()
    df_bookings["Market Area"] = df_bookings["Market Area"].str.strip()
    df_bookings["Booking Type (Low)"] = df_bookings["Booking Type (Low)"].str.strip()

    df_bookings.drop(
        columns=[
            "Bookings Hedge",
            "Market Segment",
            "Booking Type (High)",
            "Plan",
            "FX Conversion",
        ],
        inplace=True,
    )

    df_bookings.rename(
        index=str,
        columns={
            "EBU": "BU",
            "Internal Segment": "segment",
            "PMBU": "product",
            "Geo": "geo",
            "Market Area": "country",
            "Booking Type (Low)": "booking_type",
            "Value": "US_amount",
            "Fiscal Quarter": "Quarter",
        },
        inplace=True,
    )

    df_bookings.head(10)
    # df_bookings.sample(10)
    # df_bookings.tail(10)
    df_bookings = clean_bookings(df_bookings)

    return df_bookings


def clean_bookings(df_bookings):
    # ### There are new BUs now!

    # The pivot table Karen is using only look at 4 EBUs
    #  - Creative
    #  - Document Cloud
    #  - Digital Experience
    #  - Print & Publishing
    #
    #  The following bookings types are used
    #  - ASV
    #  - Total Subscription Attrition
    #  - Consulting (I do not believe this hits deferred revenue) so we drop this
    #
    #  -NOTE: As per Karen on 6/7/20, we need to add 'Premiere Support' to the ASV totals to get ours to match hers
    #
    #
    # #### This is not being done here, we have way too many different items in the 'bookings_type' field

    # ###### The cell below shows samples of what is in the data. Removing one of the parenthesis will execute the code. (One at a time)

    df_bookings["BU"].value_counts()
    # df_bookings['segment'].value_counts()
    # df_bookings['product'].value_counts()
    # df_bookings['country'].value_counts()
    # df_bookings['booking_type'].value_counts();

    change_list = [
        "Data & Insights",
        "Customer Journey Management",
        "Commerce",
        "Content",
        "Shared Marketing Cloud",
        "AEM Other",
        "Adobe  Video Solutions",
    ]

    len(change_list)

    new_BU = ["Experience Cloud"]
    new_BU_list = new_BU * len(change_list)

    change_dict = dict(zip(change_list, new_BU_list))
    print(change_dict)

    df_bookings["BU"] = df_bookings["BU"].replace(change_dict)

    df_bookings["BU"].value_counts()

    # df_bookings['BU'].value_counts()
    # df_bookings['segment'].value_counts()
    # df_bookings['product'].value_counts()
    df_bookings["country"].value_counts()
    # df_bookings['booking_type'].value_counts();

    # #### The countries now contain two fields that we need to change
    # - UNKNOWN
    # - AMER #
    #
    # These will be changed to United States

    df_bookings["country"] = df_bookings["country"].replace(
        {"AMER #": "United States", "UNKNOWN": "United States"}
    )

    df_bookings["country"].value_counts()

    df_bookings["booking_type"].value_counts()
    # df_bookings.columns

    # ##### For the booking_type we need to keep the following fields (and add them)
    # - ASV
    # - Total Subscription Attrition
    # - Premier Support (This is a new requirement for Q2 2020)
    #
    # ###### Note: These get summed by their booking amont later in the program, so we don't need to do that here

    df_bookings = df_bookings[
        df_bookings["booking_type"].isin(
            ["ASV", "Total Subscription Attrition", "Premier Support"]
        )
    ]

    df_bookings.booking_type.value_counts()

    df_bookings.tail(10)

    return df_bookings


def build_deferred_waterfall(df_billings):
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


def add_type_A_billings(billings_filename, type_A_sheetname, df, model_dict):
    temp_flat_DC, temp_flat_US = load_and_clean_type_A(
        billings_filename, model_dict, type_A_sheetname
    )

    df_billings = merge_billings_with_A(temp_flat_DC, temp_flat_US, df)

    return df_billings


def load_and_clean_type_A(
        billings_filename, model_dict, type_A_sheetname="type_A_no_config"
):
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

    df_A.head(20)

    # ##### Removing banned currencies
    # model_dict
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

    # ###### Creating a month_interval field that calculates the difference between the start_date and end_date in months. We will map this number of months into a rebilling frequency (this number of months determines when the contract expires and the deferred revenue model assumes that all attribution is accounted for in our net new billings estimates provided by FP&A)
    df_A["month_interval"] = df_A["end_date"] - df_A["start_date"]
    df_A["months"] = (df_A["month_interval"] / np.timedelta64(1, "M")).round(0)

    df_A.head(10)

    # ##### Mapping the number of months into our common rebill frequencies (monthly, quarterly, semi-annual, annual, 2 years and 3 years)
    list_rebills = [1, 3, 6, 12, 24, 36]
    temp_rebill = np.zeros_like(df_A["months"])
    for i in range(len(df_A)):
        temp_rebill[i] = min(list_rebills, key=lambda x: abs(x - df_A["months"][i]))
    df_A["rebill_months"] = temp_rebill

    fig, axs = plt.subplots(1, 1, figsize=(14, 6))
    axs.scatter(df_A["months"], df_A["rebill_months"])
    axs.set_ylabel("Rebill Months")
    axs.set_xlabel("Number of months between contract start and end dates")
    axs.set_title("Type A billings with no config type rebilling mapping")
    print_text = "No"

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

    # Quick check that we have not created duplicate column entries (for example two entries for a period with same BU and currency)
    # df_test_dup = df.copy()
    # orig_len = len(df_test_dup)
    # print("Original Length of the dataframe before duplicate test: ", orig_len)

    # df_test_dup = df_test_dup.drop_duplicates(subset=["curr", "BU", "period"])
    # print(
    #    "New length of database after duplicates have been removed: ", len(df_test_dup)
    # )

    # if orig_len != len(df_test_dup):
    #    print("We had duplicates in the dataframe! Look into why")

    return temp_flat_DC, temp_flat_US


def merge_billings_with_A(temp_flat_DC, temp_flat_US, df):
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


def load_base_billings(billings_filename, billings_sheetname="base_billings"):
    """ This loads up the base billings data and creates a dataframe

    """
    df = pd.read_excel(
        "../data/Data_2020_P06/all_billings_inputs.xlsx", sheet_name=billings_sheetname
    )
    ###### Changing the column names early since they are inconsistent across other reports
    df.rename(
        index=str,
        columns={
            "Contrct Duration in Months": "duration",
            "Document Currency": "curr",
            "Enterprise BU Desc": "BU",
            "Invoice Fiscal Year Period Desc": "period",
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

    # Removing currencies with less than n entries
    vc = df["curr"].value_counts()
    keep_these = vc.values > 20
    keep_curr = vc[keep_these]
    list_keepers = keep_curr.index
    remove_these = vc[vc.values <= 20].index
    model_dict = {"curr_removed": list(vc[remove_these].index)}
    delete_curr = list(remove_these)

    if "TRY" not in model_dict["curr_removed"]:
        model_dict["curr_removed"].append("TRY")
        delete_curr.append("TRY")
        list_keepers = list_keepers.drop("TRY")

    df = df[df["curr"].isin(list_keepers)]

    #clearing out zero amounts
    df = df[df["DC_amount"] != 0]

    # The new tableau database that uses POB type does NOT have a NON-REV type
    # Just going to comment this out
    # df = df[df["Sales Type"] != "NON-REV"]

    # ## Grouping the billings by POB Type
    # Grouping the data by the <b> Sales Type </b> field
    #  - <i>'RECOGNIZED'</i> sales are perpetual and go straight to revenue without hitting deferred
    #  - <i>'PRO-SVC-INV'</i> professional services that are invoiced and go to revenue directly when invoiced
    #  - <i>'DEFERRED'</i> sales that will sit on the balance sheet in deferred revenue and amortize over their life
    #
    #  #### Below we are creating a seperate dataframe for each of the Sales Types
    #
    list_IR = ['IR', 'IR-NA', 'LFB']
    list_service = ['CR', 'CR-NA']
    list_deferred = ['RR', 'RR-NA']
    list_hybrid = ['BNDL']
    list_all = list_IR + list_service + list_deferred + list_hybrid

    rec = df[df["POB_type"].isin(list_IR)].copy()
    svc = df[df["POB_type"].isin(list_service)].copy()
    dfr = df[df["POB_type"].isin(list_deferred)].copy()
    hyb = df[df["POB_type"].isin(list_hybrid)].copy()
    blank = df[~df["POB_type"].isin(list_all)].copy()

    # Recognized Revenue
    # Below we are grouping the rec dataframe by Currency, Business Unit and Period and cleaning up the data we do not need. Since the recognized revenue go directly to revenue, there is no contract that will renew and need to be modeled in the future.

    # testing groupby object
    gb_rec = rec.groupby(["curr", "BU", "period"], as_index=False).sum()
    gb_rec.drop(labels=["duration", "sub_term"] , axis=1, inplace=True)

    # Service Billings
    # Below we are grouping the svc dataframe by Currency, Business Unit and Period and cleaning up the data we do not need. Since the service billings go directly to revenue, there is no contract that will renew and need to be modeled in the future.
    gb_svc = svc.groupby(["curr", "BU", "period"], as_index=False).sum()
    gb_svc.drop(labels="sub_term", axis=1, inplace=True)

    # Deffered Billings
    # Type B Billings
    # THERE ARE NO LONGER ANY DEFERRED TYPE B BILLINGS!!!
    # these are now all service based billings and the POB type determines if they sit in deferred
    # such as CR or CR-NA billings or go immediately to revenue IR, IR-NA, LFB
    #dfr_b = dfr[dfr["rev_req_type"] == "B"].copy()
    #gb_b = dfr_b.groupby(["curr", "BU", "period"], as_index=False).sum()
    #gb_b.drop(labels="Subscription Term", axis=1, inplace=True)


    # #### Type A Billings
    #
    # These billings are on a billing plan. The product config tells us how long before they renew
    # and the sub term determines how often they are billed. (This is new)
    #
    #  1M: config = 'MTHLY' or sub_term = 1
    #  1Y: config = '1Y' AND sub term = 0 or 12  OR any config with sub_term = 12
    #  2Y: config = '2Y' AND sub term = 0 or 24
    #  3Y: config = '3Y' AND sub term = 0 or 36
    #  There are also config types that do not allow us to map these into a billings frequency.
    # These types are {"BLANK", "OCONS", "ONORE", "OUNIV"}
    # These types are loaded from the type_A_no_config report.

    dfr_a = dfr[dfr["rev_req_type"] == "A"].copy()

    #grouping by fields we need to keep.
    gb_a = dfr_a.groupby(["curr", "BU", "period", "config", "sub_term"], as_index=False).sum()

    print('A config value counts')
    print(gb_a["config"].value_counts(dropna=False))

    # splitting into config types we keep and ones we need to get in the type_A_no_config report
    config_type_keepers = ['MTHLY', '1Y', '2Y', '3Y']
    gb_a_keepers = gb_a[gb_a["config"].isin(config_type_keepers)].copy()
    a_blank_config = gb_a[~gb_a["config"].isin(config_type_keepers)].copy()

    print('len gb_a', len(gb_a))
    print('gb_a_keepers', len(gb_a_keepers))
    print('len a_blank_config', len(a_bad_config))
    print('Total USD Equivalent Billings of Type A with bad configs', a_blank_config.US_amount.sum())

    # ###### Grouping by the config type into gb_a_1Y, gb_a_2Y, gb_a_3y, gb_a_1M dataframes
    # Selecting monthly billings
    gb_a_1Y = gb_a_keepers[(gb_a_keepers['config'] == 'MTHLY') |
                         (gb_a_keepers['sub_term'] == 1)].copy()
    index_1M = gb_a_1Y.index
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

    print("this is the lenght of type A 1M billings: ", len(gb_a_1M))
    print("this is the lenght of type A 1Y billings: ", len(gb_a_1Y))
    print("this is the lenght of type A 2Y billings: ", len(gb_a_2Y))
    print("this is the lenght of type A 3Y billings: ", len(gb_a_3Y))


    # #### TYPE D billings
    # These billings have a field 'Rule For Bill Date' that determines when new billings will occur
    #  - Monthly:        *{Y1, Y2, Y3, Y5}*
    #  - Quarterly:      *YQ*
    #  - Every 4 months: *YT*  --NOTE: There are only 10 of these, so I am treating these as quarterly--
    #  - Semi-annual:    *YH*
    #  - Annual:         *{YA, YC}*
    #  - Every 2 years:  *Y4*
    #  - Every 3 years:  *Y7*
    #
    #  We also need to track the type D billings that do not have a 'Rule for Bill Date'

    dfr_d = dfr[dfr["rev_req_type"] == "D"].copy()

    gb_d = dfr_d.groupby(["curr", "BU", "period", "rebill_rule", "sales_doc"], as_index=False).sum()
    gb_d.drop(labels=["sub_term", "duration"], axis=1, inplace=True)

    gb_d["rebill_rule"].value_counts(dropna=False)

    # ###### Grouping these by rebill rule and incorporating rebill rules that have the same rebill period
    list_monthly = ['Y1', 'Y2', 'Y3', 'YM']
    list_qtrly = ['YQ', 'YY', 'YT']
    list_semi_ann = ['YH']
    list_ann = ['YA', 'YC', 'YX']
    list_2yrs = ['Y4']
    list_3yrs = ['Y7']
    list_all_rebills = list_monthly + list_qtrly + list_semi_ann + list_ann + list_2yrs + list_3yrs

    # TODO: We need to create a test that all of the rebill_rule fields are in list_all_rebills

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

    print("Length of monthly", len(gb_d_mthly))
    print("Length of quarterly", len(gb_d_qtrly))
    print("Length of semi ann", len(gb_d_semi_ann))
    print("Length of annual", len(gb_d_annual))
    print("Length of two years", len(gb_d_two_yrs))
    print("Length of three years", len(gb_d_three_yrs))

    # ## Building a single dataframe that incorporates all of this data
    #
    # - We will have the following descriptive fields
    #    - Invoicing Fiscal Year-Period
    #    - Document Currency Billing Amount
    #    - USD Billing Amount
    #    - Enterprise BU
    #
    # - We will have the following fields based on rebilling rule
    #    - Recognized
    #    - Service
    #    - Monthly
    #    - Quarterly
    #    - Annual
    #    - Two Years
    #    - Three Years

    # ###### Below uses functions to merge a list of dataframes and move billings amounts to the correct category based on rebill frequency and type
    #

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

    return df, model_dict


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
    ''' Testing whether we have duplicates in our merged dataframe'''
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

    df_book_period.head(14)

    df_bookings.BU.value_counts()

    # Fills in the df_book_period dataframe with the quarterly bookings numbers for each BU and currency
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

    df_book_period.tail(10)

    # ###### Cleaning up the dataframe by dropping the columns we no longer need
    df_book_period.drop(
        ["bill_Q1_sum", "bill_Q2_sum", "bill_Q3_sum", "bill_Q4_sum"], axis=1, inplace=True
    )

    df_book_period.columns

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

    # df_book_period.head(10)
    # df_book_period.sample(10)
    df_book_period.tail(10)

    # ##### The df_book_period dataframe now has columns for bookings each period in both local currency and document currency

    df_book_period.columns

    return df_book_period
