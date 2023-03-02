from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
import category_encoders as ce
from joblib import dump
from utils.PreProcessor import PreProcessor


if __name__ == '__main__':
    PreProcessingPredX_Transformer = FunctionTransformer(func = PreProcessor.PredX, check_inverse = False)
    
    labelEncoder_list = ['gender', "Partner", "Dependents", "PaperlessBilling", "PaymentMethod", "PhoneService", "InternetService", "Contract", 
                         "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    ordinalEncoder_X = ce.OrdinalEncoder(cols=labelEncoder_list)

    # without fit
    pipe_PredX = Pipeline(steps=[('preprocessing',PreProcessingPredX_Transformer), ('ordinal_y', ordinalEncoder_X)])

    dump(pipe_PredX, './src/pipe_PredX.pkl')