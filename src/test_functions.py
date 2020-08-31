def load_base_billings(config_dict):
    """ This loads up the base billings data and creates a dataframe

    """
    filename_billings = config_dict['path_to_data'] + config_dict['billings']['filename']
    sheetname_billings = config_dict['billings']['base_sheetname']
    df = pd.read_excel(filename_billings, sheet_name=sheetname_billings)

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
    # getting rid of duration here
    df.drop(labels=["duration"], axis=1, inplace=True)
    df, model_dict = clean_curr_and_zeros(df, config_dict)

    # POB Type Classifier
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

    df_hyb_IR, df_hyb_dfr = split_hybrid_dataframe(df_hyb, config_dict)

    # concatenate df_hyb_IR with rec and df_hyb_drf with df_hyb_drf
    rec = pd.concat([rec, df_hyb_IR])
    dfr = pd.concat([dfr, df_hyb_dfr])

    # Recognized Revenue
    # Below we are grouping the rec dataframe by Currency, Business Unit and Period and cleaning up the data we do not need. Since the recognized revenue go directly to revenue, there is no contract that will renew and need to be modeled in the future.

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

    # Deferred Type A Billings
    # split the type A billings based on their rebill frequency
    dfr_a = dfr[dfr["rev_req_type"] == "A"].copy()

    A_df_dict = process_type_A(config_dict, dfr_a)

    gb_a_1M = A_df_dict['gb_a_1M']
    gb_a_1Y = A_df_dict['gb_a_1Y']
    gb_a_2Y = A_df_dict['gb_a_2Y']
    gb_a_3Y = A_df_dict['gb_a_3Y']
    gb_a_no_config = A_df_dict['gb_a_no_config']

    # Deferred Type D Billings
    dfr_d = dfr[dfr["rev_req_type"] == "D"].copy()

    return_dict = process_type_D(config_dict, dfr_d)

    gb_d_mthly = return_dict['monthly']
    gb_d_qtrly = return_dict['qtrly']
    gb_d_semi_ann = return_dict['semi_ann']
    gb_d_annual = return_dict['annual']
    gb_d_two_yrs = return_dict['two_years']
    gb_d_three_yrs = return_dict['three_years']
    gb_d_no_rebill = return_dict['no_rebill']

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

