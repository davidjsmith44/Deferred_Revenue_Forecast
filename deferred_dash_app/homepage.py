"""homepage.py
Following the tutorial at https://towardsdatascience.com/create-a-multipage-dash-application-eceac464de91
"""
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

from navbar import Navbar

nav = Navbar()

# Building the body of the homepage
body = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H2("Heading"),
                        html.P(
                            """\
Donec id elit non mi porta gravida at eget metus.Fusce dapibus, tellus ac cursus commodo, tortor mauris condimentumnibh, ut fermentum massa justo sit amet risus. Etiam porta semmalesuada magna mollis euismod. Donec sed odio dui. Donec id elit nonmi porta gravida at eget metus. Fusce dapibus, tellus ac cursuscommodo, tortor mauris condimentum nibh, ut fermentum massa justo sitamet risus. Etiam porta sem malesuada magna mollis euismod. Donec sedodio dui."""
                        ),
                        dbc.Button("View details", color="secondary"),
                    ],
                    md=4,
                ),
                dbc.Col(
                    [
                        html.H2("Graph"),
                        dcc.Graph(figure={"data": [{"x": [1, 2, 3], "y": [1, 4, 9]}]}),
                    ]
                ),
            ]
        )
    ],
    className="mt-4",
)

"""Because we are building a multipage application, we need to be able to import the layout into other files.
To do this, we will build a Homepage function that returns the entire layout for the page.

Note: Layouts must always be a dash html component. The standard is to wrap the layout inside a div. """


def Homepage():
    layout = html.Div([nav, body])
    return layout


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.UNITED])

app.layout = Homepage()

if __name__ == "__main__":
    app.run_server()
