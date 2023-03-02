import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import category_encoders as ce
from joblib import dump
from utils.PreProcessor import PreProcessor


if __name__ == '__main__':
    # get training data
    train_original = pd.read_csv("./data/training_data.csv")
    # getting validation data
    val_original = pd.read_csv("./data/validation_data.csv")

    PreProcessingTrainX_Transformer = FunctionTransformer(func = PreProcessor.TrainX, check_inverse = False)
    PreProcessingTrainy_Transformer = FunctionTransformer(func = PreProcessor.Trainy, check_inverse = False)
    
    labelEncoder_list = ['gender', "Partner", "Dependents", "PaperlessBilling", "PaymentMethod", "PhoneService", "InternetService", "Contract", 
                     "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    ordinalEncoder_X = ce.OrdinalEncoder(cols=labelEncoder_list)
    ordinalEncoder_y = ce.OrdinalEncoder(cols = ['Churn'], mapping=[{'col':'Churn','mapping':{'No':0, 'Yes':1}}])

    pipe_TrainX = Pipeline(steps=[('preprocessing',PreProcessingTrainX_Transformer),('encoder_X',ordinalEncoder_X)])
    train_X = pipe_TrainX.fit_transform(train_original)
    val_X = pipe_TrainX.fit_transform(val_original)

    pipe_Trainy = Pipeline(steps=[('preprocessing',PreProcessingTrainy_Transformer), ('ordinal_y', ordinalEncoder_y)])
    train_y = pipe_Trainy.fit_transform(train_original)
    val_y = pipe_Trainy.fit_transform(val_original)

    model = RandomForestClassifier(max_depth=5, random_state=42)
    model.fit(train_X,train_y.values.ravel())

    # print(accuracy_score(model.predict(val_X), val_y.values.ravel()))
    dump(model, './src/random_forest_model.pkl')
