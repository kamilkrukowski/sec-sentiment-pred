"""Evaluate trading strategy performance over time interval."""
from typing import List, Tuple, Dict
from datetime import datetime


""" All trading strategies should have standardized input to work in automated
    downstream testing."""


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
