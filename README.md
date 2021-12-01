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

We will primarily work under the `exp.py` file (stands for experiments). Each milestone, new classifier, different and training techniques we might do should have their main execution function defined and called in `exp.py`. 

For example, we have a **Prototype 1** milestone due soon, and to run the final or one iteration of that prototype, you can call `python exp.py --exp proto1`. To add new experiments, edit the `exp.py` file.

## Existing Experiments

**Note 1:** "Test RMSLE" is evaluated on Kaggle servers as the ground truth output are not shared

**Note 2:** "Private" and "Public" are seperate Kaggle scoreboards. "Private" test data composes 51% of the total test data

| Experiment Name | Alternative Name  | Description | Validation RMSLE | Test RMSLE (Private/Public) |
| ----------- | ------------ | ----------- | ---------- | ---------- |
| Prototype 1  |  (PCA) Regression Tree  | Callable with `--exp proto1`. Executes PCA reduction from 13 to 3 dimensions in training and testing set. Experiment runs training for **Regression Tree**.       | 2.5 | 2.717/2.424 |
| Prototype 2 |  (PCA SVM Regression  | Callable with `--exp proto2`. Executes training of SVR (Support Vector Regression) from the scipy package. **This experiment is not used/included in any reports at this time.**        | N/A | N/A |
| Prototype 3 |  (PCA) kNN k=1  | Callable with `--exp proto3`. Executes fitting & evaluation of kNN (k=1) for regression. | 1.5 | 3.098/2.704 |
| Random_of_ashrae_test |  (13) Random Forest  | Callable with `randomforest_of_ashrae_test.py` Executes fitting & evaluation of Random Forest for regression. | 0.8 | 1.758/1.363 |
| Prototype 6 |  (13) MLP Keras   | Callable with `--exp proto6`. Executes training & evaluation of neural network for regression. | 2.2 | 2.306/2.239 |
| Prototype 7 |  (13) AdaBoost with Regression Tree   | Callable with `--exp adaboost_v1`. Executes fitting & evaluation of AdaBoost with Regression Trees | 4.1 | 4.559/4.129 |