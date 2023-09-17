from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


def visualisation(vis, gt_col):

    print('\n', '----------- column dtypes -----------', '\n')
    print(vis.dtypes)

    print('\n', '----------- data categories -----------', '\n')
    for column in vis.columns.values:
        if vis[column].dtype != np.int64 and vis[column].dtype != np.float64:
            print(column, vis[column].unique())

    print('\n', '----------- correlation score -----------', '\n')
    corr_matrix = vis.corr(numeric_only=True).abs()
    sol = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).stack().sort_values(ascending=False))
    print(sol)

    print('\n', '----------- rows with nan values for each column -----------', '\n')
    print(vis.isnull().sum())
    print('total rows with nan values:', vis.isnull().sum().sum())

    print('\n', '----------- number of groundtruth data -----------', '\n')
    print(vis.groupby([gt_col])[gt_col].count())

    print('\n', '-----------------------------------------', '\n')


def ten_years_old_with_thirty_years_experience(df):  # drop the data that 'person_emp_length' >= 'person_age', it does not exist in real life data

    df.drop(df[df['person_emp_length'] >= df['person_age']].index, inplace = True)  
    # l = len(df[df['person_emp_length'] >= df['person_age']].index)
    # print('dropped %d outliers' %l)

    return df


def age_preprocess(df, col):  # drop the outliers and divide ages into age groups

    df = df[df[col]<60].reset_index(drop=True)  # drop the data that age > 60
    bins= [19,25,30,35,40,45,50,55,60]
    labels = ['20-25','26-30','31-35','36-40','41-45','46-50','51-55','56-60']
    df[col] = pd.cut(df[col], bins=bins, labels=labels)

    return df


def df_label_encoder(df):

    le = preprocessing.LabelEncoder()  # transform all data into numerical data
    columns = df.columns.values
    for column in columns:
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            df[column] = le.fit_transform(df[column].astype(str))

    return df


def col_normalization(df):

    for i in df:
        df[i] = df[i] /df[i].abs().max()  # normalize all data into range of 0 to 1

    return df


def sample_dataset(df, gt_col, test_ratio):

    feature_name = df.drop([gt_col], axis=1).columns
    df_target_1 = df.loc[df[gt_col] == 1].reset_index(drop=True)
    df_target_0 = df.loc[df[gt_col] == 0].sample(n=df_target_1.shape[0])  # balance the data
    df_sampled = pd.concat([df_target_1, df_target_0], ignore_index=True)
    df_sampled = df_sampled.sample(frac=1).reset_index(drop=True)  # shuffle the data
    print(df_sampled.groupby([gt_col])[gt_col].count())

    X = df_sampled.drop([gt_col], axis=1).values
    y = df_sampled[gt_col].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio,random_state=0)

    return df_sampled, X_train, X_test, y_train, y_test


input_file = "path to credit_risk_dataset.csv"
pd.set_option('display.max_columns', None)
df = pd.read_csv(input_file, on_bad_lines='skip')
visualisation(df, gt_col='loan_status')
