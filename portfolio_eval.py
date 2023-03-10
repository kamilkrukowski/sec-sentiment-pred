"""Evaluate trading strategy performance over time interval."""
from typing import List, Tuple, Dict
from datetime import datetime


""" All trading strategies should have standardized input to work in automated
    downstream testing."""
class StockSimulation:
    def __init__(self, cash=1000000, tikrs = ['aapl','msft'], historical_data=TIKRS_dat):
        self.cash = cash
        self.tikrs = tikrs
        self.portfolio = {}
        for tikr in tikrs:
            self.portfolio[tikr] = 0
        self.active_log = []
        self.transaction_log = []
        self.transaction_cost = 0.01
        self.historical_data = TIKRS_dat
        
    def get_price(self, tikr, date):
        start = date
        if type(start) is str:
            start = datetime.strptime(start, '%Y%m%d')
          
        date = start.strftime('%Y-%m-%d')

        company_df = self.historical_data[tikr]
        return company_df[company_df['Date'] > date].iloc[0]['Open']


    def buy(self, tikr, date, allocated_money):
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
        # minus trasaction cost
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
            "balance": self.active_balance(date)
        })

    def sell(self, tikr, date, allocated_money):
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
            "balance": self.active_balance(date)
        })

    #TODO
    def get_next_trading_date(self,tikr, date):
        return None

    def rebalance(self,percentage,date):
        buy = []
        
        
        true_balance = self.active_balance(date) * 0.99
        for tikr, percent in zip(self.tikrs, percentage):
            price = self.get_price(tikr, date)
            expected_value = true_balance * percent
            current_value = self.portfolio[tikr] * price
            allocated_money = expected_value - current_value

            if allocated_money < 0:
                self.sell(tikr, date, -1 * allocated_money)
                #total_transaction += self.transaction_log[-1]['transaction_cost']
            else:
                buy += [(tikr, allocated_money )]
        
       # avg_transaction = total_transaction/len(buy)
        for tikr, allocated_money in buy:
            self.buy(tikr, date, allocated_money )#- avg_transaction)




    def active_balance(self, date):
        balance = self.cash
        for tikr in self.active_log:
            price = self.get_price(tikr, date)
            balance += self.portfolio[tikr] * price
        return balance
    
    def print_portfolio(self, date):
        balance = self.active_balance(date)
        for tikr in self.tikrs:
            price = self.get_price(tikr, date)
            tikr_holding = self.portfolio[tikr] * price
            
            print(tikr, tikr_holding/balance )
            

    def transaction_summary(self):
        for txn in self.transaction_log:
            print("{} {} {} shares of {} at ${:.2f} for ${:.2f} (transaction cost: ${:.2f}), balance: ${:.2f}".format(
                txn['date'], txn['type'], txn['shares'], txn['tikr'], txn['price'], txn['amount'], txn['transaction_cost'], txn['balance']))


def trading_strategy(
        predictions: List[Tuple(float, float, float)],
        company_list: List[str]
            ) -> Dict[datetime, List[float]]:
    """Calculate portfolio holdings at time intervals given 8-K labels."""

    strategy = dict()

    return strategy


def get_strategy_annual_return(
        strategy: Dict[datetime, List[float]],
        company_list: List[str],
        end_date: datetime,
        starting_balance: int = 1e7,
        start_date: datetime = '20000101') -> float:
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

    if type(start_date) is str:
        start_date = datetime.strptime(start_date, "%Y%m%d")
    if type(end_date) is str:
        end_date = datetime.strptime(end_date, "%Y%m%d")

    # At each date, rebalance current networth to be distributed
    # percentage-wise between companies in portfolio_allocations

    return 1.00
