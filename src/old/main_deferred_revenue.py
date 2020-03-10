'''
This is the first attempt to change the deferred revenue model to be easier to use in python (so that anyone can run it)


'''



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import Counter, defaultdict



df = pd.read_excel('inputs/base_billings.xlsx', sheet_name='bill_DC')

# removing currencies that have less than 10 transactions in them
dc_vc = df['Document Currency'].value_counts()
# boolean mask on the vc values (the index is the currency itself)
keep_these = dc_vc.values > 10
keep_curr = dc_vc[keep_these]
a = keep_curr.index
df = df[df['Document Currency'].isin(a)]

#clearing out any zeros in the document currency billings
df = df[df['Completed Sales Doc Currency']!=0]


'''
Options for how to best do this

# want to use groupby and split apply combine somehow

problem: I have multiple splits to perform here

BU
Currency
Time
Billings Type




class?

What is the object???




'''

class SingleBUCurr(object):

    def __init__(self):
        self.BU =
        self.currency =
        self.
        pass


    def forecast(self, n_periods):

        pass

    def __repr__(self):


class BusinessUnit(object):

    def __init__(self):

        pass

    def

''' start with the most basic unit you can think of


This would be recognized revenue
    dates
    currency
    bu
'''

class CurrencyBillings(object)"
"
class RecognizedRevenue(object):


    def __init__(self):

        pass
class ServiceRevenue(object):
    def __init(self):
        pass

class DeferredBilling(object):
    def __init__(self):
        pass

# do any of these methods need a forecast method?????

'''
This is a mess!


Make a class for Currency

currency dictionary???
WTF else???
'''


# Trying a dictionary of dictionaries

dict_curr = {'EUR': eur_dict,
             'USD': usd_dict,
             'JPY': jpy_dict
             }



eur_dict =

''' How the hell am I going to do this with different currencies '''

