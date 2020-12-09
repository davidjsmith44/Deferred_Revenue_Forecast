def load_bookings(config_dict):
    '''
    This function creates a dataframe that contains the bookings forecast that FP&A provides.

    :param config_dict: This is the dictionary that contains all of the parameters for the deferred model
    :return:
    '''
    filename_bookings = config_dict['path_to_data'] + config_dict['bookings']['filename']
    bookings_sheetname = config_dict['bookings']['sheetname']
    df_bookings = pd.read_excel(filename_bookings, bookings_sheetname)

    # Cleaning up the bookings data
    # NOTE: The bookings spreadsheet looks very different for Q2 versus prior quarters!
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

    df_bookings = clean_bookings(df_bookings)

    return df_bookings


def clean_bookings(df_bookings):
    '''
    The pivot table Karen is using only look at 4 EBUs
      - Creative
      - Document Cloud
      - Digital Experience
      - Print & Publishing

      The following bookings types are used
      - ASV
      - Total Subscription Attrition
      - Consulting (I do not believe this hits deferred revenue) so we drop this

      -NOTE: As per Karen on 6/7/20, we need to add 'Premiere Support' to the ASV totals to get ours to match hers


    :param df_bookings:
    :return:
    '''

    # #### This is not being done here, we have way too many different items in the 'bookings_type' field

    # ###### The cell below shows samples of what is in the data. Removing one of the parenthesis will execute the code. (One at a time)


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
    # ###### Note: These get summed by their booking amount later in the program, so we don't need to do that here
    #
    # New bookings file is just Net Asv, so we no longer need to include permiere support
    df_bookings = df_bookings[
        df_bookings["booking_type"].isin(
            ["Net ASV"]
        )
    ]

    return df_bookings
