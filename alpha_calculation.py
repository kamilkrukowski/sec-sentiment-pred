import dataloading as dl
import numpy as np
import pandas as pd


INPUT_FILE = 'test_output_test1.csv'
PICKLED_YFINANCE = 'TIKR_DATA.pkl'
OUTPUT_FILE = 'test_output_test1_with_alpha.csv'

data = pd.read_csv(INPUT_FILE, index_col = 0, parse_dates = ["Date"])

# filter out prediction  == 0
data = data[data.pred== 1]
yd = dl.HistoricalYahoo.load(PICKLED_YFINANCE)

data = data[data.tikr == "ABBV"]

data['Release Date'] = data['Date']

# adding annual return information from TIKR_DATA.pkl
dl.get_reference_data(data, yd, cols=[
        'Date', 'Outlook', 'Percent Return', 'Annual Return', 'sp Annual', 'sp Percent', 'beta'],
        backfill=False)




# Loading the risk free rate information
rf_info = pd.read_csv("Risk_free_rate.csv", parse_dates = ["Date"], index_col = "Date")

# calculate the product once and pass it as an argument to the function
rf_cumprod = (1+rf_info/100).cumprod() 
rf_date, cumprod_rf  =  rf_info.index, rf_cumprod.values.tolist()


# based on risk free rate calculate 90 days alpha values
data["jensen alpha (90)"] = data.progress_apply(lambda x: dl.calculate_jensen_alpha(x, rf_date, cumprod_rf, inname = "Percent Return",  hold_period = 90), axis=1, result_type='expand')
data["simple alpha (90)"] = data.progress_apply(lambda x: dl.calculate_simple_alpha(x, rf_date, cumprod_rf, inname = "Percent Return",  hold_period = 90), axis=1, result_type='expand')


data.to_csv(OUTPUT_FILE)