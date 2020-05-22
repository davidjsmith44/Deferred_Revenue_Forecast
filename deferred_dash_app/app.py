""" app.py """

### Data
import pandas as pd
import pickle

#Graphing
import plotly.graph_objects as go

### Dash
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input
## Navbar
from navbar import Navbar

df = pd.read_csv('https://gist.githubusercontent.com/joelsewhere/f75da35d9e0c7ed71e5a93c10c52358d/raw/d8534e2f25495cc1de3cd604f952e8cbc0cc3d96/population_il_cities.csv')
df.set_index(df.iloc[:,0], drop = True, inplace = True)
df = df.iloc[:,1:]

nav = Navbar()

header = html.H3(' Select the name of an Illinois city to see its population"')

'''
Dropdown:
For this application we will be building a dropdown menu. In the background, dropdown menus are formatted as a list of dictionaries.
This dictionary has two keys. “label” and “value”. label = what the user will see in the dropdown menu,
 and value = what you are returning to the application to query the data.
Since each column is an individual city in Illinois, we will be using the column names of this dataset for querying.
Every column is formatted as “City, Illinois”. Having the “, Illinois” included for every label is a bit overkill
since we are only looking cities from Illinois. In the code below we will remove this extra bit of
text for the label of each dictionary:
'''
#Just removing the state since they are all in Illinois
options = [{'label':x.replace(', Illinois', ''), 'value': x} for x in df.columns]

'''
IMPORTANT: Any component that is interactive must have an id name. You’ll see why soon.
Additionally, components cannot share id-names. If an id is used more than once,
an error will be thrown and the application will break.
The parameters in the dcc.Dropdown function are.
  id: A unique identifier for a component. Formatted as a string.
  options: The list of dictionaries with “label” and “value” keys
  value: A default value that the dropdown is set to when the application loads.
  (I chose a random city from the dataset for this default.)
'''