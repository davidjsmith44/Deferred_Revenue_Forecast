
import pandas as pd
import numpy as np

df = pd.read_excel("data/Data_2020_P06/Q2'20 Rev Acctg Mgmt Workbook (06-04-20).xlsx",
                   sheet_name='Deferred Revenue Forecast', skiprows=5)

df.head(50)