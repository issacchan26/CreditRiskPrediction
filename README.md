# Credit Risk Prediction with CatBoostClassifier
Data Analysis and prediction on Kaggle dataset: Credit Risk Dataset  
This repo provides algorithms for data preprocessing and model training for Credit Risk problem  
CatBoostClassifier is used as prediction model in this repo  
Dataset: [https://www.kaggle.com/datasets/laotse/credit-risk-dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)

## Data Preprocessing
Before running [preprocess.py](preprocess.py), please change the path of 'credit_risk_dataset.csv' to your local path  
It provides data visualisation of the dataset, also including the preprocess functions before training the model  

## Model Training
Before running [train.py](train.py), please change the path of 'credit_risk_dataset.csv' to your local path  
This repo used CatBoostClassifier as prediction model  
Default device is GPU, please comment the line of 'task_type="GPU",' if using CPU

## Performance
Most of the method in Kaggle are using full dataset for training and evaluation dataset  
After considering the unbalance on ground truth may lead to bias on the model, 
This repo will resample and balance the groundtruth data where number of (groundtruth == 1) equals to number of (groundtruth == 0)  
Currently the model can reach ~86.81 of accuracy in balanced dataset, while reach ~93% in full dataset



