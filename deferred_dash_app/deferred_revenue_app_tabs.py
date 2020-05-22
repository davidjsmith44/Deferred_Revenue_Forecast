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

df_wf = import_thing["wf"]
# print("historical billings length: ", len(df_billings))
# print("forecast length: ", len(df_fcst))
# print("combined length: ", len(df))
# print("length of df_wf: ", len(df_wf))
# print(df_wf.columns)
# print(df_wf.head(10))

list_currencies = df["curr"].unique()
list_BUs = df["BU"].unique()
df["monthly_periods"] = df["deferred_1M_DC"] / df["Period_Weeks"]


# adding the app itself (all dash apps will havae this line of code. It initiates the code)
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


app.layout = html.Div(
    [
        html.H4(children="Deferred Revenue Forecast", style={"text-align": "center"}),
        dcc.Tabs(
            [
                dcc.Tab(label="Sunburst Charts", children=[]),
                dcc.Tab(
                    label="Document Currency Billings",
                    children=[
                        html.Div(
                            [
                                dcc.Dropdown(
                                    id="currency_DC",
                                    options=[
                                        {"label": i, "value": i}
                                        for i in list_currencies
                                    ],
                                    value="USD",
                                ),
                                dcc.RadioItems(
                                    id="BU_DC",
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
                                    dcc.Graph(id="deferred_1Y_DC"),
                                    className="six columns",
                                ),
                                html.Div(
                                    dcc.Graph(id="deferred_1M_DC"),
                                    className="six columns",
                                ),
                            ],
                            className="row",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    dcc.Graph(id="deferred_2Y_DC"),
                                    className="six columns",
                                ),
                                html.Div(
                                    dcc.Graph(id="deferred_6M_DC"),
                                    className="six columns",
                                ),
                            ],
                            className="row",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    dcc.Graph(id="deferred_3Y_DC"),
                                    className="six columns",
                                ),
                                html.Div(
                                    dcc.Graph(id="deferred_3M_DC"),
                                    className="six columns",
                                ),
                            ],
                            className="row",
                        ),
                        html.Div(
                            html.Div(
                                dcc.Graph(id="deferred_all_DC"),
                                className="twelve columns",
                            )
                        ),
                    ],
                ),
                dcc.Tab(
                    label="USD Equivalent Billings",
                    children=[
                        html.Div(
                            [
                                dcc.Dropdown(
                                    id="currency_US",
                                    options=[
                                        {"label": i, "value": i}
                                        for i in list_currencies
                                    ],
                                    value="USD",
                                ),
                                dcc.RadioItems(
                                    id="BU_US",
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
                                    dcc.Graph(id="deferred_1Y_US"),
                                    className="six columns",
                                ),
                                html.Div(
                                    dcc.Graph(id="deferred_1M_US"),
                                    className="six columns",
                                ),
                            ],
                            className="row",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    dcc.Graph(id="deferred_2Y_US"),
                                    className="six columns",
                                ),
                                html.Div(
                                    dcc.Graph(id="deferred_6M_US"),
                                    className="six columns",
                                ),
                            ],
                            className="row",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    dcc.Graph(id="deferred_3Y_US"),
                                    className="six columns",
                                ),
                                html.Div(
                                    dcc.Graph(id="deferred_3M_US"),
                                    className="six columns",
                                ),
                            ],
                            className="row",
                        ),
                        html.Div(
                            html.Div(
                                dcc.Graph(id="deferred_all_US"),
                                className="twelve columns",
                            )
                        ),
                    ],
                ),
                dcc.Tab(
                    label="Deferred Revenue Forecast",
                    children=[
                        html.Div(
                            [
                                dcc.Dropdown(
                                    id="currency_WF",
                                    options=[
                                        {"label": i, "value": i}
                                        for i in list_currencies
                                    ],
                                    value="USD",
                                ),
                                dcc.RadioItems(
                                    id="BU_WF",
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
                                    dcc.Graph(id="this_waterfall"),
                                    className="twelve columns",
                                ),
                            ],
                            className="row",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    dcc.Graph(id="BU_waterfall"),
                                    className="tweleve columns",
                                ),
                            ],
                            className="row",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    dcc.Graph(id="total_waterfall"),
                                    className="twelve columns",
                                ),
                            ],
                            className="row",
                        ),
                    ],
                ),
            ]
        ),
    ]
)


@app.callback(
    Output("deferred_3Y_DC", "figure"),
    [Input("currency_DC", "value"), Input("BU_DC", "value")],
)
def update_3Y_graph_DC(currency_value, BU_value):
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
    Output("deferred_2Y_DC", "figure"),
    [Input("currency_DC", "value"), Input("BU_DC", "value")],
)
def update_2Y_graph_DC(currency_value, BU_value):
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
    Output("deferred_1Y_DC", "figure"),
    [Input("currency_DC", "value"), Input("BU_DC", "value")],
)
def update_1Y_graph_DC(currency_value, BU_value):
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
    Output("deferred_6M_DC", "figure"),
    [Input("currency_DC", "value"), Input("BU_DC", "value")],
)
def update_6M_graph_DC(currency_value, BU_value):
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
    Output("deferred_3M_DC", "figure"),
    [Input("currency_DC", "value"), Input("BU_DC", "value")],
)
def update_3M_graph_DC(currency_value, BU_value):
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
    Output("deferred_1M_DC", "figure"),
    [Input("currency_DC", "value"), Input("BU_DC", "value")],
)
def update_1M_graph_DC(currency_value, BU_value):
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
    Output("deferred_all_DC", "figure"),
    [Input("currency_DC", "value"), Input("BU_DC", "value")],
)
def update_all_graphs_DC(currency_value, BU_value):
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


@app.callback(
    Output("deferred_3Y_US", "figure"),
    [Input("currency_US", "value"), Input("BU_US", "value")],
)
def update_3Y_graph_US(currency_value, BU_value):
    dff = df[(df["BU"] == BU_value) & (df["curr"] == currency_value)]
    this_length = len(dff)
    colors = ["lightgrey"] * len(dff)
    change_list = np.arange(this_length - 48, this_length - 36)
    for item in change_list:
        colors[item] = "dimgrey"
    y_title = "USD Eqivalent 3 Year Billings from " + str(currency_value)
    return {
        "data": [
            {
                "x": dff["period"].to_list(),
                "y": dff["deferred_3Y_US"].to_list(),
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
    Output("deferred_2Y_US", "figure"),
    [Input("currency_US", "value"), Input("BU_US", "value")],
)
def update_2Y_graph_US(currency_value, BU_value):
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
    y_title = "USD Equivalent if 2 Year Billings in " + str(currency_value)
    return {
        "data": [
            {
                "x": dff["period"],
                "y": dff["deferred_2Y_US"],
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
    Output("deferred_1Y_US", "figure"),
    [Input("currency_US", "value"), Input("BU_US", "value")],
)
def update_1Y_graph_US(currency_value, BU_value):
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
                "y": dff["deferred_1Y_US"],
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
    Output("deferred_6M_US", "figure"),
    [Input("currency_US", "value"), Input("BU_US", "value")],
)
def update_6M_graph_US(currency_value, BU_value):
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
                "y": dff["deferred_6M_US"],
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
    Output("deferred_3M_US", "figure"),
    [Input("currency_US", "value"), Input("BU_US", "value")],
)
def update_3M_graph_US(currency_value, BU_value):
    dff = df[(df["BU"] == BU_value) & (df["curr"] == currency_value)]
    colors = ["darkviolet"] * len(dff)
    this_length = len(dff["period"])
    change_list = np.arange(this_length - 15, this_length - 12)
    for item in change_list:
        colors[item] = "violet"
    y_title = "USD Equivalent of Quarterly Billings in " + str(currency_value)
    return {
        "data": [
            {
                "x": dff["period"],
                "y": dff["deferred_3M_US"],
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
    Output("deferred_1M_US", "figure"),
    [Input("currency_US", "value"), Input("BU_US", "value")],
)
def update_1M_graph_US(currency_value, BU_value):
    dff = df[(df["BU"] == BU_value) & (df["curr"] == currency_value)]
    colors = ["cornflowerblue"] * len(dff)
    y_title = "Monthly Billings in " + str(currency_value)
    return {
        "data": [
            {
                "x": dff["period"],
                "y": dff["deferred_1M_US"],
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
    Output("deferred_all_US", "figure"),
    [Input("currency_US", "value"), Input("BU_US", "value")],
)
def update_all_graphs_US(currency_value, BU_value):
    dff = df[(df["BU"] == BU_value) & (df["curr"] == currency_value)]
    return {
        "data": [
            {
                "x": dff["period"].to_list(),
                "y": dff["deferred_3Y_US"].to_list(),
                "type": "bar",
                "marker": {"color": "dimgrey"},
                "name": "3 year",
            },
            {
                "x": dff["period"].to_list(),
                "y": dff["deferred_2Y_US"].to_list(),
                "type": "bar",
                "marker": {"color": "burleywood"},
                "name": "2 year",
            },
            {
                "x": dff["period"].to_list(),
                "y": dff["deferred_1Y_US"].to_list(),
                "type": "bar",
                "marker": {"color": "darkgreen"},
                "name": "1 year",
            },
            {
                "x": dff["period"].to_list(),
                "y": dff["deferred_6M_US"].to_list(),
                "type": "bar",
                "marker": {"color": "salmon"},
                "name": "6 month",
            },
            {
                "x": dff["period"].to_list(),
                "y": dff["deferred_3M_US"].to_list(),
                "type": "bar",
                "marker": {"color": "darkviolet"},
                "name": "quarterly",
            },
            {
                "x": dff["period"].to_list(),
                "y": dff["deferred_1M_US"].to_list(),
                "type": "bar",
                "marker": {"color": "cornflowerblue"},
                "name": "monthly",
            },
            {
                "x": dff["period"].to_list(),
                "y": dff["deferred_B_US"].to_list(),
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


@app.callback(
    Output("this_waterfall", "figure"),
    [Input("currency_WF", "value"), Input("BU_WF", "value")],
)
def update_this_waterfall(currency_value, BU_value):
    df_wff = df_wf[(df_wf["BU"] == BU_value) & (df_wf["curr"] == currency_value)]
    colors = ["cornflowerblue"] * len(df_wff)
    y_title = "Deferred Balance in USD"
    return {
        "data": [
            {
                "x": df_wff["period"],
                "y": df_wff["Total"],
                "type": "bar",
                "marker": {"color": colors},
                "name": "Total Deferred Balance",
            },
        ],
        "layout": dict(
            xaxis={"title": "Fiscal Period"},
            yaxis={"title": y_title},
            transition={"duration": 500, "easing": "cubic-in-out"},
            legend=dict(x=0.1, y=0.9),
            title="Deferred Balance in USD for billings in "
            + str(currency_value)
            + " from "
            + str(BU_value),
            hovermode="closest",
        ),
    }


@app.callback(
    Output("BU_waterfall", "figure"),
    [Input("currency_WF", "value"), Input("BU_WF", "value")],
)
def update_BU_waterfall(currency_value, BU_value):
    df_wff = df_wf.drop(["curr"], axis=1)
    df_wf2 = df_wff[df_wff["BU"] == BU_value]
    print("I am in update_BU_waterfall")
    print(df_wf2.head(10))
    df_wf3 = df_wf2.groupby(["period"]).sum()
    # need to groupby here

    colors = ["cornflowerblue"] * len(df_wf3)
    y_title = "Deferred Balance in USD"
    return {
        "data": [
            {
                # "x": df_wf3["period"],
                "y": df_wf3["Total"],
                "type": "bar",
                "marker": {"color": colors},
                "name": "Deferred Balance in USD for " + str(BU_value) + ".",
            },
        ],
        "layout": dict(
            xaxis={"title": "Fiscal Period"},
            yaxis={"title": y_title},
            transition={"duration": 500, "easing": "cubic-in-out"},
            legend=dict(x=0.1, y=0.9),
            title="Deferred Balance in USD for "
            + str(BU_value)
            + " billings in all currencies.",
            hovermode="closest",
        ),
    }


@app.callback(
    Output("total_waterfall", "figure"),
    [Input("currency_WF", "value"), Input("BU_WF", "value")],
)
def update_total_waterfall(currency_value, BU_value):
    df_wff = df_wf.drop(columns=["curr", "BU"], axis=1)
    df_wff = df_wff.groupby(["period"]).agg("sum")
    # need to do a groupby here
    colors = ["cornflowerblue"] * len(df_wff)
    y_title = "Deferred Balance in USD"
    return {
        "data": [
            {
                # "x": df_wff["period"],
                "y": df_wff["Total"],
                "type": "bar",
                "marker": {"color": colors},
                "name": y_title,
            },
        ],
        "layout": dict(
            xaxis={"title": "Fiscal Period"},
            yaxis={"title": y_title},
            transition={"duration": 500, "easing": "cubic-in-out"},
            legend=dict(x=0.1, y=0.9),
            title="Deferred Balance in USD for all Adobe",
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
