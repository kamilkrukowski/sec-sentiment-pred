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
