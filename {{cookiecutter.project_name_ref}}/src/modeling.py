"""Modeling class and functions."""

from typing import Any
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from src.preprocessing import fillna_median


class Classify:
    """
    A classification pipeline that handles preprocessing and classification
    tasks.

    This class automatically handles categorical and missing data
    preprocessing, and fits a specified model to the transformed data.
    """

    def __init__(self, df: pd.DataFrame, target: str, model: Any) -> None:
        """
        Initialize the Classify object with data, target, and model
        to be used.

        Args:
            df (pd.DataFrame): The input data frame containing features and
            target.
            target (str): The name of the target variable in the data frame.
            model (Any): A scikit-learn-compatible model that implements the
            fit and predict methods.
        """
        self.df = df
        self.target = target
        self.model = model
        self.cat_cols = df.select_dtypes(include=object).columns.to_list()
        self.with_nan_cols = df.columns[df.isnull().any().to_list()].to_list()
        self.fillna_transformer = self.build_fillna_transformer()
        self.cat_transformer = self.build_cat_transformer()
        self.pipeline = self.build_pipeline()

    def build_fillna_transformer(self) -> FunctionTransformer:
        """
        Construct a transformer for filling missing values in the data frame.

        Returns:
            FunctionTransformer: A transformer that fills missing values using
            the median computed by group.
        """
        fillna_transformer = FunctionTransformer(
            fillna_median,
            kw_args={'grp': self.cat_cols, 'cols': self.with_nan_cols}
        )
        return fillna_transformer

    def build_cat_transformer(self) -> ColumnTransformer:
        """
        Construct a transformer to convert categorical variables into
        dummy/one-hot encoded variables.

        Returns:
            ColumnTransformer: A transformer that applies one-hot encoding to
            categorical columns.
        """
        cat_transformer = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.cat_cols)
            ],
            remainder='passthrough'
        )
        return cat_transformer

    def build_pipeline(self) -> Pipeline:
        """
        Build the complete preprocessing and classification pipeline.

        Returns:
            Pipeline: A machine learning pipeline that includes median filling,
            categorical encoding, robust scaling, and classification.
        """
        pipeline = Pipeline(steps=[
            ('fill_median', self.fillna_transformer),
            ('preprocessor', self.cat_transformer),
            ('robust_scaler', RobustScaler()),
            ('classifier', self.model)
        ])
        return pipeline

    def get_pipeline(self) -> Pipeline:
        """Get the last pipeline built.

        Returns:
            Pipeline: Last pipeline built.
        """
        return self.pipeline

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Fit the pipeline to the training data.

        Args:
            x_train (pd.DataFrame): The training feature data.
            y_train (pd.Series): The training target data.
        """
        self.pipeline.fit(
            x_train, y_train
        )

    def predict(self, x_test: pd.DataFrame) -> pd.Series:
        """
        Predict the target for the test data.

        Args:
            x_test (pd.DataFrame): The test feature data.

        Returns:
            ndarray: The predicted classes.
        """
        pred = self.pipeline.predict(x_test)
        return pred

    def predict_proba(self, x_test: pd.DataFrame) -> pd.Series:
        """
        Predict class probabilities for the test data.

        Args:
            x_test (pd.DataFrame): The test feature data.

        Returns:
            ndarray: The predicted class probabilities.
        """
        pred = self.pipeline.predict_proba(x_test)
        return pred
