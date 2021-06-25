



import pandas as pd
import pickle
import openpyxl


output_dict = pickle.load(open("/Volumes/Treasury/Financial_Database/Deferred_Revenue/Inputs/Data_2020_p12/processed/final_forecast3.p", "rb"))

print(output_dict.keys)
