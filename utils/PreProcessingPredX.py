
def PreProcessingPredX(df):
    df = df[df['TotalCharges'] != ' ']
    df['TotalCharges'] = df['TotalCharges'].astype(float)
    # drop customer ID: not a feature for training 
    df = df.drop('customerID', axis=1)
    df = df.drop(columns='Churn')
    return df