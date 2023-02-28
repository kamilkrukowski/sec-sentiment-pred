# sec-sentiment-pred

In this repository, we create a bag-of-words model that predicts whether 8-K documents forecast rise or fall in stock price

## Dependencies

We use [EDGAR-DOC-PARSER](https://kamilkrukowski.github.io/EDGAR-DOC-PARSER) to create a dataset.
It is currently available on the PyPi Test Server


### Conda

To create conda environment:
```
conda create -n sectagging -c conda-forge -c anaconda python=3.10 pytorch scipy numpy selenium=4.5.0 pyyaml chardet requests lxml scikit-learn pandas pytorch
conda activate sectagging
pip install -i https://test.pypi.org/simple/ edgar-doc-parser==0.1.7
pip install secedgar==0.4.0
```
