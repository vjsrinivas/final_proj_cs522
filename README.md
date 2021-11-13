# COSC522 Final Project
## ASHRAE Great Energy Predictor Challenge

**Team Members:** Ravi Patel, Jay Pike, Vijay Rajagopal, Samuel Okhuegbe, and Chris O’Brien

## Preparing Environment

### Data Prep

To create the same environment as this GitHub repo, please visit the [Kaggle reference](https://www.kaggle.com/c/ashrae-energy-prediction/data?select=train.csv), and download all present csv files. Create a folder called "ashrae" under the "data" folder (i.e. `mkdir ./data/ashrae`). Place the csv files under the "./data/ashrae" folder.

### Python Environment

Everything is run on Python 3.6 (specifically 3.6.12), but 3.8 to 3.5 _should_ work, but no promises. The following packages are needed:

* numpy
* scipy
* sklearn
* matplotlib
* pandas
* tqdm

For an automated installation under a majority `pip` environment, you can use: `pip install -r requirements.txt`

## Running Experiments

We will primarily work under the `exp.py` file (stands for experiments). Each milestone, new classifier, different and training techniques we might do should have their main execution function defined and called in `exp.py`. For example,we have a **Prototype 1** milestone due soon, and to run the final or one iteration of that prototype, you can call `python exp.py --exp proto1`. To add new experiments, edit the `exp.py` file.

