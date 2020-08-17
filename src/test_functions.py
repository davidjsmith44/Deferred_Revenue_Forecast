
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
    gb_rec.drop(labels=["duration", "sub_term"] , axis=1, inplace=True)

    # Service Billings
    # Below we are grouping the svc dataframe by Currency, Business Unit and Period and
    # cleaning up the data we do not need. Since the service billings go directly to revenue,
    # there is no contract that will renew and need to be modeled in the future.
    # TODO: Take care of the service billings (how should they amortize?)
    # possibly only a problem when we get to deferred revenue (forecasting)
    gb_svc = svc.groupby(["curr", "BU", "period"], as_index=False).sum()
    gb_svc.drop(labels=["sub_term", "duration"], axis=1, inplace=True)

    dfr.drop(labels=['duration'], axis=1, inplace=True)

    # Deferred Type A Billings
    # split the type A billings based on their rebill frequency
    dfr_a = dfr[dfr["rev_req_type"] == "A"].copy()

    A_df_dict = process_type_A(config_dict, dfr_A)
    gb_a_1M = A_df_dict['gb_a_1M']
    gb_a_1Y = A_df_dict['gb_a_1Y']
    gb_a_2Y = A_df_dict['gb_a_2Y']
    gb_a_3Y = A_df_dict['gb_a_3Y']
    gb_a_no_config = A_df_dict['gb_a_no_config']

    # Deferred Type D Billings
    dfr_d = dfr[dfr["rev_req_type"] == "D"].copy()

    return_dict = process_type_D(config_dict, df)

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


def clean_curr_and_zeros(df, config_dict):
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

    # clearing out zero amounts
    df = df[df["DC_amount"] != 0]

    return df, model_dict

def process_type_A(config_dict, dfr_A):
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

    :param dfr_A: This is the dataframe of deferred billings that have revenue recognition type A
    :return: return_A_dict: A dictionary containing dataframes containing the billings that have the same
                rebill frequency.
                return_A_dict = {'gb_a_1M': gb_a_1M,
                     'gb_a_1Y': gb_a_1Y,
                     'gb_a_2Y': gb_a_2Y,
                     'gb_a_3Y': gb_a_3Y,
                     'gb_a_no_config': gb_a_no_config,}
    '''

    # grouping by fields we need to keep.
    gb_a = dfr_a.groupby(["curr", "BU", "period", "config", "sub_term"], as_index=False).sum()

    print('A config value counts')
    print(gb_a["config"].value_counts(dropna=False))

    # splitting into config types we keep and ones we need to get in the type_A_no_config report
    config_type_keepers = config_dict['type_A_config_keepers']
    gb_a_keepers = gb_a[gb_a["config"].isin(config_type_keepers)].copy()
    df_a_no_config = gb_a[~gb_a["config"].isin(config_type_keepers)].copy()

    print('len gb_a', len(gb_a))
    print('gb_a_keepers', len(gb_a_keepers))
    print('len df_a_no_config', len(df_a_no_config))
    print('Total USD Equivalent Billings of Type A with bad configs', df_a_no_config.US_amount.sum())

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

    # dropping duration from the gb_a_#X below here
    return_A_dict = {'gb_a_1M': gb_a_1M,
                     'gb_a_1Y': gb_a_1Y,
                     'gb_a_2Y': gb_a_2Y,
                     'gb_a_3Y': gb_a_3Y,
                     'gb_a_no_config': gb_a_no_config,}
    return return_A_dict


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


