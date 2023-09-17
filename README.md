# Kaggle dataset: Credit Risk Dataset
Data Analysis and prediction on Kaggle dataset: Credit Risk Dataset  
This repo provides algorithms for data preprocessing and model training for Credit Risk problem  
Dataset: [https://www.kaggle.com/datasets/laotse/credit-risk-dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)

## Data Preprocessing
Before running [preprocess.py](preprocess.py), please change the path of 'credit_risk_dataset.csv' to your local path  
It provides data visualisation of the dataset, also including the preprocess functions before training the model  

## Model Training
Please change the path of 'credit_risk_dataset.csv' to your local path  
This repo used CatBoostClassifier as prediction model  
Default device is GPU, please comment the line of 'task_type="GPU",' if using CPU



