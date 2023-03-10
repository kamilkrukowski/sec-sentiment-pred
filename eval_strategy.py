from . import strategies
from portfolio_eval import get_strategy_annual_return

strategy = strategies.baseline

labels = [int(i.split('\t')[1]) for i in open('train.tsv').readlines()]
company_list = open('tickrs.txt', 'r').read().strip().split('\n')

allocations = strategy(labels, company_list)

print(get_strategy_annual_return(allocations, company_list))