# sec-sentiment-pred

In this repository, we create a bag-of-words model that predicts whether 8-K documents forecast rise or fall in stock price

## Dependencies

We use [EDGAR-DOC-PARSER](https://kamilkrukowski.github.io/EDGAR-DOC-PARSER) to create a dataset.

### Conda

To create conda environment:
```
conda create -n sectagging -c conda-forge -c anaconda python=3.10 pytorch scipy numpy selenium=4.5.0 pyyaml chardet requests lxml scikit-learn pandas pytorch
conda activate sectagging
pip install edgar-doc-parser==0.2.2.post1
```
## Pipeline

Use ```generate_data.py``` to make ```8k_data_raw.tsv```
Use ```filter.py``` on ```8k_data_raw.tsv``` to make ```8k_data_filtered.tsv```
Use ```dataloading.py``` on ```8k_data_filtered.tsv``` to make ```8k_data.tsv```, the final version

For test runs, use ```generate_short.py``` on ```8k_data.tsv``` to make ```8k_data_short.tsv```


## Alpha Calculation

1. download Risk_free_rate.csv 

2. To add alpha into data
```
	#adding beta
    get_reference_data(data, yd, cols=['beta', "Percent Return (1)"])
    rf_info = pd.read_csv("Risk_free_rate.csv", parse_dates = ["Date"], index_col = "Date")
    # calculate the sum once and pass it as an argument to the function
    rf_cumsum = rf_info.cumsum()
    rf_date, cumsum_rf  =  rf_info.index, rf_cumsum.values.tolist()

    # adding alpha
    data["jensen alpha (90)"] = data.progress_apply(lambda x: calculate_jensen_alpha(x, rf_date, cumsum_rf,inname = "Percent Return",  hold_period = 90), axis=1, result_type='expand')
    data["simple alpha (90)"] = data.progress_apply(lambda x: calculate_simple_alpha(x, rf_date, cumsum_rf,inname = "Percent Return",  hold_period = 90), axis=1, result_type='expand')
    data["jensen alpha (1)"] = data.progress_apply(lambda x: calculate_jensen_alpha(x, rf_date, cumsum_rf,inname = "Percent Return (1)",  hold_period = 1), axis=1, result_type='expand')
    data["simple alpha (1)"] = data.progress_apply(lambda x: calculate_simple_alpha(x, rf_date, cumsum_rf,inname = "Percent Return (1)",  hold_period = 1), axis=1, result_type='expand')
    print(data)
```