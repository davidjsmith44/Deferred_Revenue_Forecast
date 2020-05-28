import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import pickle

import plotly.express as px

# import plotly.offline as pyo
import plotly.graph_objs as go

import_thing = pickle.load(open("../data/processed/final_forecast.p", "rb"))
df_fcst = import_thing["forecast"]
df_billings = import_thing["billings"]
df_billings["is_forecast"] = 0
df_fcst["is_forecast"] = 1

df = pd.concat([df_billings, df_fcst], join="outer", ignore_index=True)
df = df.fillna(0)
df.sort_values(by=["curr", "BU", "period"], inplace=True)

df["period"] = df["period"].astype("string")

df_2019 = df[df["period"].str.match("2019")]

df2 = (
    df_2019.set_index(["curr", "BU", "period"])
    .stack()
    .reset_index(name="Val")
    .rename(columns={"level_1": "X"})
)


df_2019_US = df2[df2["level_3"].str.contains("_US")].copy()
df_2019_US["curr"] = df_2019_US["curr"].astype("string")
df_2019_US["BU"] = df_2019_US["BU"].astype("string")
df_2019_US["period"] = df_2019_US["period"].astype("string")
df_2019_US["level_3"] = df_2019_US["level_3"].astype("string")
df_2019_US.rename(columns={"level_3": "type"}, inplace=True)

df_2019_gb = df_2019_US.groupby(["curr", "BU", "type"]).sum()

df_2019_gb.reset_index(inplace=True)
df_2019_gb["base"] = "test"
print(df_2019_gb.head(10))
print(len(df_2019_gb))

df_2019_gb = df_2019_gb[df_2019_gb["Val"] > 0]

fig = px.sunburst(df_2019_gb, path=["BU", "curr", "type"], values="Val")
fig.show()
