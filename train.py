from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from preprocess import ten_years_old_with_thirty_years_experience, age_preprocess, df_label_encoder, col_normalization, sample_dataset

input_file = "path to credit_risk_dataset.csv"
pd.set_option('display.max_columns', None)
df = pd.read_csv(input_file, on_bad_lines='skip')

df = ten_years_old_with_thirty_years_experience(df)
df = df.dropna()  # Not replace the nan values with medians/modes for best prediction result
df = age_preprocess(df, 'person_age')
df = df_label_encoder(df)
df = col_normalization(df)
df_balanced, X_train, X_test, y_train, y_test = sample_dataset(df, gt_col='loan_status', test_ratio=0.1)  # will print the balanced ground truth dataframe

model = CatBoostClassifier(
    iterations=4400,
    task_type="GPU",  # comment this line if using CPU
    learning_rate=0.01,
    random_seed=0
)

model.fit(X_train, y_train, 
        eval_set=(X_test, y_test), 
        verbose=True
)

print(model.score(X_test, y_test))  # accuracy
