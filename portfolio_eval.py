"""Evaluate trading strategy performance over time interval."""
from typing import List, Tuple, Dict
from datetime import datetime
import pickle

""" All trading strategies should have standardized input to work in automated
    downstream testing."""
class StockSimulation:
    def __init__(self, historical_data, cash=1000000,  tikrs = ['aapl','msft']):

        """
        Initialize a StockSimulation instance.

        Parameters
        ----------
        cash : float
            The initial amount of cash to invest, by default 1000000
        tikrs : List[str]
            The list of stock symbols to invest in, by default ['aapl', 'msft']
        historical_data : Dict[str, pd.DataFrame]
            A dictionary of historical stock data for each stock symbol in `tikrs`.
            Each key is a stock symbol, and each value is a pandas DataFrame containing the
            historical data.
        Returns
        -------
        None
        """

        self.cash = cash
        self.tikrs = tikrs
        self.portfolio = {}
        for tikr in tikrs:
            self.portfolio[tikr] = 0
        self.active_log = []
        self.transaction_log = []
        self.transaction_cost = 0.01
        self.historical_data = historical_data
        
    def get_price(self, tikr, date):
        """
        Get the opening price of a stock on a given date.

        Parameters
        ----------
        tikr : str
            The stock symbol to get the price for.
        date : str or datetime
            The date to get the price for. If str, must be in the format 'YYYYMMDD'.

        Returns
        -------
        float
            The opening price of the stock on the given date.
        """

        start = date
        if type(start) is str:
            start = datetime.strptime(start, '%Y%m%d')
          
        date = start.strftime('%Y-%m-%d')

        company_df = self.historical_data[tikr]
        return company_df[company_df['Date'] > date].iloc[0]['Open']


    def buy(self, tikr, date, allocated_money):
        """
        Buy shares of a stock.

        Parameters
        ----------
        tikr : str
            The stock symbol to buy.
        date : str or datetime
            The date to buy the stock. If str, must be in the format 'YYYYMMDD'.
        allocated_money : float
            The amount of money to allocate to the stock.

        Returns
        -------
        None
        """
        # check if there is enough cash to buy
        if self.cash < allocated_money:
            print("Not enough cash to buy")
            return

        # calculate number of shares to buy
        price = self.get_price(tikr, date)
        shares = allocated_money / price

        # update portfolio and cash balance
        if tikr in self.active_log:
            self.portfolio[tikr] += shares
        else:
            self.portfolio[tikr] = shares
            self.active_log.append(tikr)
        # minus the cost of stock
        stock_cost = shares * price
        #self.cash -= stock_cost * (1 + self.transaction_cost)
        self.cash -= stock_cost 


        # log transaction
        self.transaction_log.append({
            "type": "buy",
            "tikr": tikr,
            "date": date,
            "shares": shares,
            "price": price,
            "amount": stock_cost ,
            "transaction_cost": 0,
            "net_worth": self.get_net_worth(date)
        })

    def sell(self, tikr, date, allocated_money):
        """
        Sell shares of a stock.

        Parameters
        ----------
        tikr : str
            The stock symbol to sell.
        date : str or datetime
            The date to sell the stock. If str, must be in the format 'YYYYMMDD'.
        allocated_money : float
            The amount of money to allocate to the stock.

        Returns
        -------
        None
        """
        # check if there are enough shares to sell
        if tikr not in self.portfolio:
            print("No shares of {} in portfolio".format(tikr))
            return


        # calculate number of shares to sell
        price = self.get_price(tikr, date)
        shares = allocated_money / price


        if self.portfolio[tikr] < shares:
            print("Not enough shares of {} to sell".format(tikr))
            return

        # update portfolio and cash balance
        self.portfolio[tikr] -= shares
        if self.portfolio[tikr] == 0:
            self.active_log.remove(tikr)

        # plus the earning
        stock_cost = shares * price
        self.cash += stock_cost
        # minus trasaction cost
        self.cash -= stock_cost *  self.transaction_cost


        # log transaction
        self.transaction_log.append({
            "type": "sell",
            "tikr": tikr,
            "date": date,
            "shares": shares,
            "price": price,
            "amount": stock_cost ,
            "transaction_cost": stock_cost * self.transaction_cost,
            "net_worth": self.get_net_worth(date)
        })

    # #TODO
    # def get_next_trading_date(self,tikr, date):
        
    #     return None

    def rebalance(self, percentage, date):
        """
        Rebalance the portfolio according to a target allocation.

        Parameters
        ----------
        percentage : List[float]
            The target allocation percentages for each stock in the portfolio.
        date : str or datetime
            The date to rebalance the portfolio. If str, must be in the format 'YYYYMMDD'.

        Returns
        -------
        None
        """

        buy = []
        
        # We multiply the net worth by 0.99 to account for transaction
        # costs incurred during selling. Specifically, the
        # `sell()` method has a transaction cost of 1 percent, 
        # so we reduce the net worth by 1% to account for this cost.#
        true_balance = self.get_net_worth(date) * 0.99

        for tikr, percent in zip(self.tikrs, percentage):
            price = self.get_price(tikr, date)
            expected_value = true_balance * percent
            current_value = self.portfolio[tikr] * price
            allocated_money = expected_value - current_value

            if allocated_money < 0:
                self.sell(tikr, date, -1 * allocated_money)
            else:
                buy += [(tikr, allocated_money )]

        for tikr, allocated_money in buy:
            self.buy(tikr, date, allocated_money )




    def get_net_worth(self, date):
        """
        Calculates the net worth of the portfolio on a given date, including
        cash and holdings of all active stocks in the portfolio.
        
        Parameters
        ----------
        date : datetime.datetime or str
            The date on which to calculate the active balance of the portfolio.
            If str, the accepted format is "year_month_day".
            
        Returns
        -------
        balance : float
            The net worth of the portfolio on the given date.
        """
        balance = self.cash
        for tikr in self.active_log:
            price = self.get_price(tikr, date)
            balance += self.portfolio[tikr] * price
        return balance
    
    def print_portfolio(self, date):
        """
        Prints the percentage of the portfolio holdings that are invested in each
        active stock in the portfolio, based on the net worth on the given date.
        
        Parameters
        ----------
        date : datetime.datetime or str
            The date on which to calculate the active balance of the portfolio.
            If str, the accepted format is "year_month_day".
            
        Returns
        -------
        None
        """
        net_worth = self.get_net_worth(date)
        print("portfolio_allocation on", date)
        for tikr in self.tikrs:
            price = self.get_price(tikr, date)
            tikr_holding = self.portfolio[tikr] * price
            
            print(tikr, tikr_holding/net_worth)
        print('cash', self.cash)
        print()
            

    def transaction_summary(self):
        """
        Prints a summary of all transactions made in the portfolio, including
        the date, type, number of shares, stock ticker, price per share,
        transaction amount, transaction cost, and remaining balance.
        
        Returns
        -------
        None
        """
        for txn in self.transaction_log:
            print("{} {} {} shares of {} at ${:.2f} for ${:.2f} (transaction cost: ${:.2f}), balance: ${:.2f}".format(
                txn['date'], txn['type'], txn['shares'], txn['tikr'], txn['price'], txn['amount'], txn['transaction_cost'], txn['net_worth']))

"""
def trading_strategy(
        predictions: List[tuple],
        company_list: List[str]
            ) -> Dict[datetime, List[float]]:
    #Calculate portfolio holdings at time intervals given 8-K labels.

    strategy = dict()

    return strategy
"""

def load_historical_data(filename):
    with open(filename, 'rb') as handle:
        tikr_dict = pickle.load(handle)
        return tikr_dict

def get_strategy_annual_return(
        strategy: Dict[datetime, List[float]],
        company_list: List[str],
        end_date: datetime,
        starting_balance: int = 1e7,
        start_date: datetime = '20000101' ,
        silence = True) -> float:
    """
    Calculate annual return over time period.
    
    Parameters
    ----------
    strategy: Dict[datetime, List[float]]
        A dictionary of portfolio rebalance dates associated with portfolio
        allocation percentages. The portfolio allocation percentages share the
        same indexing as company_list.
    company_list: List[str]
        The companies invested into by the trading strategy percent allocation.
    start_date: datetime, str
        Accepted format "year_month_day", or datetime object. The first day of
        trading.
    end_date: datetime, str
        Accepted format "year_month_day", or datetime object. The final day of
        trading used to determine net worth and annualized return.
    starting_balance: int
        The initial amount of money invested.

    Returns
    -------
    Annualized Return: float
        The annualized return rate, where 0% return indicates 1.00.
    """
    TIKRS_dat = load_historical_data('TIKR_DATA.pickle')

    if type(start_date) is str:
        start_date = datetime.strptime(start_date, "%Y%m%d")
    if type(end_date) is str:
        end_date = datetime.strptime(end_date, "%Y%m%d")

    s = StockSimulation(TIKRS_dat, cash = starting_balance, tikrs = company_list)
    for date, portfolio_allocation in strategy.items():
        # At each date, rebalance current networth to be distributed
        # percentage-wise between companies in portfolio_allocations
        s.rebalance(percentage= portfolio_allocation, date = date)
        if not silence:
            s.print_portfolio(date)
    if not silence:
        s.transaction_summary()

    # Calculate the number of days between the start and end dates
    delta_days = (end_date - start_date).days

    # Calculate the fractional number of years using the total number of days
    n = delta_days / 365.25  # assuming a leap year every 4 years   


    return (s.get_net_worth(end_date)/starting_balance) \
            ** ( 1 / n)  - 1
