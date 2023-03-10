import portfolio_eval
import pickle
#import generate_labels

def load_historical_data(filename):
    with open(filename, 'rb') as handle:
        tikr_dict = pickle.load(handle)
        return tikr_dict
TIKRS_dat = load_historical_data('TIKR_DATA.pickle')


#s = portfolio_eval.StockSimulation(TIKRS_dat,cash =1000)
#s.rebalance([0.5,0.5],'20001210')
#s.print_portfolio( '20001210')
#s.rebalance([0.3,0.7], "20011210")
#s.print_portfolio( "20011210")
#s.rebalance([0,0],'20100112')

company_list = ['aapl', 'msft']
end_date = "20100112"
start_date = '20001210'
starting_balance = 1000
strategy = {"20001210": [0.5,0.5], "20011210": [0.3,0.7],"20100112": [0,0] }

perc_change = portfolio_eval.get_strategy_annual_return(strategy,company_list,end_date,starting_balance, start_date)
print("percent change: " , perc_change)