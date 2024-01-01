import sys 
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import save_object
from src.exception import CustomException
from src.logger import logging


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transform_obj(self, numerical_cols, categorical_cols):
        try:
            
            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )
            logging.info('numeric col transformation done')

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )
            logging.info('categorical col transformation done')

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline, numerical_cols),
                    ("cat_pipeline",cat_pipeline, categorical_cols)
                ]
            )
            logging.info('combined the cat and num pipelines')

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transform(self, train_data_path, test_data_path):
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            logging.info("read train and test data")

            target_column_name = "math_score"
            
            #train data
            feature_train_df = train_df.drop(columns = [target_column_name], axis = 1)
            target_train_df = train_df[target_column_name]
            #test data
            feature_test_df = test_df.drop(columns = [target_column_name], axis = 1)
            target_test_df = test_df[target_column_name]

            numeric_features = [feature for feature in feature_train_df.columns if feature_train_df[feature].dtype != 'O'][1:] # numberic
            categorical_features = [feature for feature in feature_train_df.columns if feature_train_df[feature].dtype == 'O'] # catergory
            preprocessor_obj = self.get_data_transform_obj(numeric_features, categorical_features)


            #conv df to array after preprocessing
            # fit transform will learn the parameters for the dataset(mean, s.d. while transform applies the same parameters for transforming which was learned during fit_transform) 
            preprocessed_feature_train_array = preprocessor_obj.fit_transform(feature_train_df)
            preprocessed_feature_test_array = preprocessor_obj.transform(feature_test_df)

            #concatinating the feature array and target array
            train_arr = np.c_[preprocessed_feature_train_array, np.array(target_train_df)]
            test_arr = np.c_[preprocessed_feature_test_array, np.array(target_test_df)]

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e,sys)