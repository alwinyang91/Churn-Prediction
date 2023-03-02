import joblib
import pandas as pd
from utils.PreProcessor import PreProcessor


single_X_original = pd.read_csv('./data/single_row_to_check.csv')
pipe_PredX = joblib.load("./src/pipe_PredX.pkl")
single_X = pipe_PredX.fit_transform(single_X_original)

model = joblib.load("./src/random_forest_model.pkl")
pred_y = model.predict(single_X)

if pred_y == 0:
    print('Customer will not churn.')
else:
    print('Customer will churn.')