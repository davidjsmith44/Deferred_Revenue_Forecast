""" deferred_revenue_app """

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

import pandas as pd
import numpy as np
import pickle


external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

# loading up my data from deferred revenue
import_thing = pickle.load(open("data/final_forecast.p", "rb"))
df_fcst = import_thing["forecast"]
df_billings = import_thing["billings"]
df_billings["is_forecast"] = 0
df_fcst["is_forecast"] = 1

df = pd.concat([df_billings, df_fcst], join="outer", ignore_index=True)
df = df.fillna(0)
df.sort_values(by=["curr", "BU", "period"], inplace=True)

print("historical billings length: ", len(df_billings))
print("forecast length: ", len(df_fcst))
print("combined length: ", len(df))

list_currencies = df["curr"].unique()
list_BUs = df["BU"].unique()
df["monthly_periods"] = df["deferred_1M_DC"] / df["Period_Weeks"]


# adding the app itself (all dash apps will havae this line of code. It initiates the code)
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


app.layout = html.Div(
    [
        html.H4(children="Deferred Revenue Forecast"),
        dcc.Tabs(
            [
                dcc.Tab(label="Sunburst Charts", childeren=[]),
                dcc.Tab(
                    label="Document Currency Billings",
                    childern=[
                        html.Div(
                            [
                                dcc.Dropdown(
                                    id="currency",
                                    options=[
                                        {"label": i, "value": i}
                                        for i in list_currencies
                                    ],
                                    value="USD",
                                ),
                                dcc.RadioItems(
                                    id="BU",
                                    options=[
                                        {"label": i, "value": i} for i in list_BUs
                                    ],
                                    value="Creative",
                                    labelStyle={"display": "inline-block"},
                                ),
                            ],
                            style={"width": "48%", "display": "inline-block"},
                        ),
                        html.Div(
                            [
                                html.Div(
                                    dcc.Graph(id="deferred_1Y"), className="six columns"
                                ),
                                html.Div(
                                    dcc.Graph(id="deferred_1M"), className="six columns"
                                ),
                            ],
                            className="row",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    dcc.Graph(id="deferred_2Y"), className="six columns"
                                ),
                                html.Div(
                                    dcc.Graph(id="deferred_6M"), className="six columns"
                                ),
                            ],
                            className="row",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    dcc.Graph(id="deferred_3Y"), className="six columns"
                                ),
                                html.Div(
                                    dcc.Graph(id="deferred_3M"), className="six columns"
                                ),
                            ],
                            className="row",
                        ),
                        html.Div(
                            html.Div(
                                dcc.Graph(id="deferred_all"), className="twelve columns"
                            )
                        ),
                    ],
                ),
                dcc.Tab(label="USD Equivalent Billings", children=[]),
                dcc.Tab(label="Deferred Revenue Forecast", children=[]),
            ]
        ),
    ]
)


@app.callback(
    Output("deferred_3Y", "figure"), [Input("currency", "value"), Input("BU", "value")],
)
def update_3Y_graph(currency_value, BU_value):
    dff = df[(df["BU"] == BU_value) & (df["curr"] == currency_value)]
    this_length = len(dff)
    colors = ["lightgrey"] * len(dff)
    change_list = np.arange(this_length - 48, this_length - 36)
    for item in change_list:
        colors[item] = "dimgrey"
    y_title = "2 Year Billings in " + str(currency_value)
    return {
        "data": [
            {
                "x": dff["period"].to_list(),
                "y": dff["deferred_3Y_DC"].to_list(),
                "type": "bar",
                "marker": {"color": colors},
                "name": "Deferred Billings with 3 year Billings",
            }
        ],
        "layout": dict(
            xaxis={"title": "Fiscal Period"},
            yaxis={"title": y_title},
            transition={"duration": 500, "easing": "cubic-in-out"},
            title="3 Year Billing Cycle",
            hovermode="closest",
        ),
    }


@app.callback(
    Output("deferred_2Y", "figure"), [Input("currency", "value"), Input("BU", "value")],
)
def update_2Y_graph(currency_value, BU_value):
    dff = df[(df["BU"] == BU_value) & (df["curr"] == currency_value)]
    this_length = len(dff)
    colors = ["burlywood"] * len(dff)
    line_colors = ["burlywood"] * len(dff)
    change_list = np.arange(this_length - 36, this_length - 24)
    for item in change_list:
        colors[item] = "chocolate"
    line_change_list = np.arange(this_length - 11, this_length)
    for item in line_change_list:
        line_colors[item] = "chocolate"
    y_title = "2 Year Billings in " + str(currency_value)
    return {
        "data": [
            {
                "x": dff["period"],
                "y": dff["deferred_2Y_DC"],
                "type": "bar",
                "marker": {"color": colors},
                "marker_line": {"color": line_colors},
                "name": "Is this working",
            }
        ],
        "layout": dict(
            xaxis={"title": "Fiscal Period"},
            yaxis={"title": y_title},
            transition={"duration": 500, "easing": "cubic-in-out"},
            title="2 Year Billings Cycle",
            hovermode="closest",
        ),
    }


@app.callback(
    Output("deferred_1Y", "figure"), [Input("currency", "value"), Input("BU", "value")],
)
def update_1Y_graph(currency_value, BU_value):
    dff = df[(df["BU"] == BU_value) & (df["curr"] == currency_value)]
    this_length = len(dff)
    colors = ["darkgreen"] * len(dff)
    change_list = np.arange(this_length - 24, this_length - 12)
    for item in change_list:
        colors[item] = "lightgreen"
    y_title = "Annual Billings in " + str(currency_value)
    return {
        "data": [
            {
                "x": dff["period"],
                "y": dff["deferred_1Y_DC"],
                "type": "bar",
                "marker": {"color": colors},
                "name": "Deferred Annual Billings",
            }
        ],
        "layout": dict(
            xaxis={"title": "Fiscal Period"},
            yaxis={"title": y_title},
            transition={"duration": 500, "easing": "cubic-in-out"},
            title="Annual Billing Cycle",
            hovermode="closest",
        ),
    }


@app.callback(
    Output("deferred_6M", "figure"), [Input("currency", "value"), Input("BU", "value")],
)
def update_6M_graph(currency_value, BU_value):
    dff = df[(df["BU"] == BU_value) & (df["curr"] == currency_value)]
    this_length = len(dff)
    colors = ["salmon"] * len(dff)
    change_list = np.arange(this_length - 18, this_length - 12)
    for item in change_list:
        colors[item] = "crimson"
    y_title = "Semi-Annual Billings in " + str(currency_value)
    return {
        "data": [
            {
                "x": dff["period"],
                "y": dff["deferred_6M_DC"],
                "type": "bar",
                "marker": {"color": colors},
                "name": "Deferred Semi-Annual Billings",
            }
        ],
        "layout": dict(
            xaxis={"title": "Fiscal Period"},
            yaxis={"title": y_title},
            transition={"duration": 500, "easing": "cubic-in-out"},
            title="Semi-Annual Billings",
            hovermode="closest",
        ),
    }


@app.callback(
    Output("deferred_3M", "figure"), [Input("currency", "value"), Input("BU", "value")],
)
def update_3M_graph(currency_value, BU_value):
    dff = df[(df["BU"] == BU_value) & (df["curr"] == currency_value)]
    colors = ["darkviolet"] * len(dff)
    this_length = len(dff["period"])
    change_list = np.arange(this_length - 15, this_length - 12)
    for item in change_list:
        colors[item] = "violet"
    y_title = "Quarterly Billings in " + str(currency_value)
    return {
        "data": [
            {
                "x": dff["period"],
                "y": dff["deferred_3M_DC"],
                "type": "bar",
                "marker": {"color": colors},
                "name": "Deferred Quarterly Billings",
            }
        ],
        "layout": dict(
            xaxis={"title": "Fiscal Period"},
            yaxis={"title": y_title},
            transition={"duration": 500, "easing": "cubic-in-out"},
            title="Quarterly Billings",
            hovermode="closest",
        ),
    }


@app.callback(
    Output("deferred_1M", "figure"), [Input("currency", "value"), Input("BU", "value")]
)
def update_1M_graph(currency_value, BU_value):
    dff = df[(df["BU"] == BU_value) & (df["curr"] == currency_value)]
    colors = ["cornflowerblue"] * len(dff)
    y_title = "Monthly Billings in " + str(currency_value)
    return {
        "data": [
            {
                "x": dff["period"],
                "y": dff["deferred_1M_DC"],
                "type": "bar",
                "marker": {"color": colors},
                "name": "Deferred Monthly Billings",
            },
            {
                "x": dff["period"],
                "y": dff["monthly_periods"],
                "type": "line",
                "marker_color": "purple",
                "name": "Weekly Avg",
                "seconday_y": True,
                "range": [0, dff["monthly_periods"].max()],
            },
        ],
        "layout": dict(
            xaxis={"title": "Fiscal Period"},
            yaxis={"title": y_title},
            transition={"duration": 500, "easing": "cubic-in-out"},
            legend=dict(x=0.1, y=0.9),
            title="Monthly Billings",
            hovermode="closest",
        ),
    }


@app.callback(
    Output("deferred_all", "figure"),
    [Input("currency", "value"), Input("BU", "value")],
)
def update_all_graphs(currency_value, BU_value):
    dff = df[(df["BU"] == BU_value) & (df["curr"] == currency_value)]
    return {
        "data": [
            {
                "x": dff["period"].to_list(),
                "y": dff["deferred_3Y_DC"].to_list(),
                "type": "bar",
                "marker": {"color": "dimgrey"},
                "name": "3 year",
            },
            {
                "x": dff["period"].to_list(),
                "y": dff["deferred_2Y_DC"].to_list(),
                "type": "bar",
                "marker": {"color": "burleywood"},
                "name": "2 year",
            },
            {
                "x": dff["period"].to_list(),
                "y": dff["deferred_1Y_DC"].to_list(),
                "type": "bar",
                "marker": {"color": "darkgreen"},
                "name": "1 year",
            },
            {
                "x": dff["period"].to_list(),
                "y": dff["deferred_6M_DC"].to_list(),
                "type": "bar",
                "marker": {"color": "salmon"},
                "name": "6 month",
            },
            {
                "x": dff["period"].to_list(),
                "y": dff["deferred_3M_DC"].to_list(),
                "type": "bar",
                "marker": {"color": "darkviolet"},
                "name": "quarterly",
            },
            {
                "x": dff["period"].to_list(),
                "y": dff["deferred_1M_DC"].to_list(),
                "type": "bar",
                "marker": {"color": "cornflowerblue"},
                "name": "monthly",
            },
            {
                "x": dff["period"].to_list(),
                "y": dff["deferred_B_DC"].to_list(),
                "type": "bar",
                "marker": {"color": "yellow"},
                "name": "service",
            },
        ],
        "layout": dict(
            xaxis={"title": "Fiscal Period"},
            yaxis={"title": "All Deferred Billings"},
            transition={"duration": 500, "easing": "cubic-in-out"},
            title="All Deferred Billings",
            barmode="stack",
            hovermode="closest",
        ),
    }


if __name__ == "__main__":
    app.run_server(debug=True)

    # loading up my data from deferred revenue
    # import_thing = pickle.load(open('../data/processed/initial_forecast.p', 'rb'))
    # df_fcst = import_thing['forecast']
    # df_billings = import_thing['billings']
    # df_billings['is_forecast']= 0
    # df_fcst['is_forecast']=1
    # df = pd.concat([df_billings, df_fcst],
    #            join='outer',
    #            ignore_index=True)
    # df = df.fillna(0)
    # df.sort_values(by=['curr', 'BU', 'period'], inplace=True)

    # this_slice = df[(df['curr']=='USD')&
    #                (df['BU']=='Creative')]
