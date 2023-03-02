import pandas as pd

class PreProcessor:

    def __init__(self):
        pass
        

    def PredX(df):
        df = df[df['TotalCharges'] != ' ']
        df['TotalCharges'] = df['TotalCharges'].astype(float)
        # drop customer ID: not a feature for training 
        df =  df.drop('customerID', axis=1)
        df =  df.drop(columns='Churn')
        return  df

    def TrainX(df):
        df = df[df['TotalCharges'] != ' ']
        df['TotalCharges'] = df['TotalCharges'].astype(float)
        # drop customer ID: not a feature for training 
        df_trainX = df.drop('customerID', axis=1)
        df_trainX = df_trainX.drop(columns='Churn')
        return df_trainX

    def Trainy(df):
        df = df[df['TotalCharges'] != ' ']
        df_trainy = pd.DataFrame(df['Churn'])
        return df_trainy