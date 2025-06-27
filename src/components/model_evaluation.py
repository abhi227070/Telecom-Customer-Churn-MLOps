from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact, DataTransformationArtifact
from sklearn.metrics import accuracy_score
from src.exception import CustomException as MyException
from src.constants import TARGET_COLUMN
from src.logger import logging
from src.utils.main_utils import load_object
import sys
import pandas as pd
from typing import Optional
from src.entity.s3_estimator import Proj1Estimator
from dataclasses import dataclass

@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    difference: float


class ModelEvaluation:

    def __init__(self, model_eval_config: ModelEvaluationConfig, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_artifact: DataTransformationArtifact,model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise MyException(e, sys) from e

    def get_best_model(self) -> Optional[Proj1Estimator]:
        """
        Method Name :   get_best_model
        Description :   This function is used to get model from production stage.
        
        Output      :   Returns model object if available in s3 storage
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path=self.model_eval_config.s3_model_key_path
            proj1_estimator = Proj1Estimator(bucket_name=bucket_name,
                                               model_path=model_path)

            if proj1_estimator.is_model_present(model_path=model_path):
                return proj1_estimator
            return None
        except Exception as e:
            raise  MyException(e,sys)
        
    def _change_dtype(self, df):
        """Change the data type of 'TotalCharges' column to float."""
        logging.info("Changing 'TotalCharges' column data type to float")
        
        change_col = 'TotalCharges'
        
        if change_col in df.columns:
            df[change_col] = pd.to_numeric(df[change_col], errors='coerce')
            
        return df
    
    def _drop_column(self, df):
        """Drop the 'id' column if it exists."""
        logging.info("Dropping 'id' column")
        drop_col = 'customerID'
        if drop_col in df.columns:
            df = df.drop(drop_col, axis=1)
        return df

    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Method Name :   evaluate_model
        Description :   This function is used to evaluate trained model 
                        with production model and choose best model 
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            x, y = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]

            logging.info("Test data loaded and now transforming it for prediction...")
            
            encoder = load_object(file_path=self.data_transformation_artifact.transformed_encoder_file_path)

            x = self._change_dtype(x)
            x = self._drop_column(x)

            trained_model = load_object(file_path=self.model_trainer_artifact.trained_model_file_path)
            
            
            y = encoder.transform(y)
            
            logging.info("Trained model loaded/exists.")
            trained_model_accuracy = self.model_trainer_artifact.metric_artifact.accuracy_score
            trained_model_output = trained_model.predict(x)
            trained_model_accuracy = accuracy_score(y, trained_model_output)
            logging.info(f"Accuracy_Score for this model: {trained_model_accuracy}")

            best_model_accuracy_score=None
            best_model = self.get_best_model()
            if best_model is not None:
                logging.info(f"Computing F1_Score for production model..")
                y_hat_best_model = best_model.predict(x)
                best_model_accuracy_score = accuracy_score(y, y_hat_best_model)
                logging.info(f"Accuracy-Production Model: {best_model_accuracy_score}, Accuracy-New Trained Model: {trained_model_accuracy}")
            
            tmp_best_model_score = 0 if best_model_accuracy_score is None else best_model_accuracy_score
            result = EvaluateModelResponse(trained_model_f1_score=trained_model_accuracy,
                                           best_model_f1_score=best_model_accuracy_score,
                                           is_model_accepted=trained_model_accuracy > tmp_best_model_score,
                                           difference=trained_model_accuracy - tmp_best_model_score
                                           )
            logging.info(f"Result: {result}")
            return result

        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function is used to initiate all steps of the model evaluation
        
        Output      :   Returns model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """  
        try:
            print("------------------------------------------------------------------------------------------------")
            logging.info("Initialized Model Evaluation Component.")
            evaluate_model_response = self.evaluate_model()
            s3_model_path = self.model_eval_config.s3_model_key_path

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                s3_model_path=s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference)

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise MyException(e, sys) from e