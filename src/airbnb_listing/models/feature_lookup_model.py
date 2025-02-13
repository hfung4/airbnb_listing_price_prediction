from datetime import datetime

import mlflow
import pandas as pd
from databricks import feature_engineering
from databricks.connect import DatabricksSession
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from databricks.sdk import WorkspaceClient
from lightgbm import LGBMRegressor
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from airbnb_listing.config import Config, Tags
from airbnb_listing.logging import logger


# Feature Lookup Model
class FeatureLookUpModel:
    def __init__(self, config: Config, tags: Tags, spark: DatabricksSession):
        """initialize the FeatureLookUpModel class

        Args:
            config (Config): configuration object
            tags (Tags): tag object
            spark (DatabricksSession): spark session
        """
        self.config = config
        self.tags = tags
        self.spark = spark
        self.workspace = WorkspaceClient()
        self.fe = feature_engineering.FeatureEngineeringClient()

        # Get configuration variables
        self.num_features = self.config.model.SELECTED_NUMERIC_FEATURES
        self.cat_features = self.config.model.SELECTED_CATEGORICAL_FEATURES
        self.ID_COLUMN = self.config.model.ID_COLUMN
        self.target = self.config.model.TARGET
        self.parameters = self.config.model.MODEL_PARAMS
        self.catalog_name = (
            self.config.general.DEV_CATALOG
        )  # hardcoded for now, later it will be dependent on the target environment
        self.silver_schema = self.config.general.SILVER_SCHEMA
        self.gold_schema = self.config.general.GOLD_SCHEMA
        self.ml_asset_schema = self.config.general.ML_ASSET_SCHEMA

        # Define the feature table name and feature function name
        self.feature_table_name = f"{self.catalog_name}.{self.gold_schema}.{self.config.general.FEATURE_TABLE_NAME}"
        self.function_name = f"{self.catalog_name}.{self.gold_schema}.calculate_date_since_last_review"

        # Mlflow configuration
        self.experiment_name = self.config.general.EXPERIMENT_NAME_FE
        # self.tags = tags.model_dump()
        self.tags = tags.dict()

    def create_feature_table(self):
        """Create the feature table in the gold layer."""

        query = f"""
            CREATE OR REPLACE TABLE {self.feature_table_name} (
            {self.ID_COLUMN} STRING NOT NULL,
            latitude DOUBLE,
            longitude DOUBLE,
            is_manhattan BOOLEAN
        );
        """
        self.spark.sql(query)

        # Set primary key
        self.spark.sql(
            f"ALTER TABLE {self.feature_table_name} ADD CONSTRAINT listing_pk PRIMARY KEY ({self.ID_COLUMN});"
        )
        # Set table properties to enable change data feed (needed for creating online tables)
        self.spark.sql(f"ALTER TABLE {self.feature_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")

        # Insert train and test set into the feature table to ensure we can perform feature lookup for
        # ALL listings (either used for training and testing). If there is an unknonw/new listing, then
        # we cannot perform feature lookup, and therefore we cannot make predictions for it.

        # Insert train set
        query = f""" INSERT INTO {self.feature_table_name}
         SELECT {self.ID_COLUMN}, latitude, longitude, is_manhattan FROM
            {self.catalog_name}.{self.silver_schema}.airbnb_listing_price_train"""
        self.spark.sql(query)

        # Insert test set
        query = f""" INSERT INTO {self.feature_table_name}
         SELECT {self.ID_COLUMN}, latitude, longitude, is_manhattan FROM
            {self.catalog_name}.{self.silver_schema}.airbnb_listing_price_test"""
        self.spark.sql(query)

        logger.info("✅ Feature table created and populated.")

    # Feature function definition
    def create_feature_function(self):
        """Define a function to compute date since last review"""
        query = f"""
        CREATE OR REPLACE FUNCTION {self.function_name}(last_review TIMESTAMP)
        RETURNS DOUBLE
        LANGUAGE PYTHON AS
        $$
        from datetime import datetime, timezone
        if last_review is None:
            return None
        else:
            return (datetime.now(timezone.utc) - last_review).total_seconds() / 86400
        $$
        """
        self.spark.sql(query)

    # Load silver train and test data
    def load_silver_data(self):
        """Load train and test data from the silver layer"""
        self.train_set = self.spark.table(f"{self.catalog_name}.{self.silver_schema}.airbnb_listing_price_train").drop(
            "latitude", "longitude", "is_manhattan"
        )
        self.train_set = self.train_set.withColumn("last_review", self.train_set["last_review"].cast("timestamp"))

        # Since I need the test set to evaluate the trained model and compute performance metrics, I need
        # all features (including the ones that will be retrieved from the feature table)
        self.test_set = self.spark.table(
            f"{self.catalog_name}.{self.silver_schema}.airbnb_listing_price_test"
        ).toPandas()

        logger.info("✅ Data loaded successfully.")

    # Create training set with features from 1) silver train set, 2) feature table, and 3) feature function
    def feature_engineering(self):
        """Perform feature engineering by linking silver ata with feature tables"""
        # Create the specification for the training set
        self.training_set_spec = self.fe.create_training_set(
            df=self.train_set,
            label=self.target,
            feature_lookups=[
                # If I want to use multiple feature tables, I can add additional FeatureLookup objects
                FeatureLookup(
                    table_name=self.feature_table_name,
                    feature_names=["latitude", "longitude", "is_manhattan"],
                    lookup_key=self.ID_COLUMN,
                ),
                FeatureFunction(
                    udf_name=self.function_name,
                    output_name="days_since_last_review",
                    # key is input argument of the function, value is the column name in input dataframe
                    input_bindings={"last_review": "last_review"},
                ),
            ],
            exclude_columns=["update_timestamp_utc"],
        )

        # Create the training set (in Pandas)
        self.training_df = self.training_set_spec.load_df().toPandas()

        # Create the days_since_last_review feature for the test set that is used
        # to evaluate the trained model
        self.test_set["days_since_last_review"] = self.test_set["last_review"].apply(
            lambda x: (datetime.now() - x).days if pd.notna(x) else None
        )

        # Create X_train, y_train, X_test, y_test for model training and evaluation
        self.X_train = self.training_df[self.num_features + self.cat_features + ["days_since_last_review"]]
        self.y_train = self.training_df[self.target]
        self.X_test = self.test_set[self.num_features + self.cat_features]
        self.X_test = self.test_set[self.num_features + self.cat_features + ["days_since_last_review"]]
        self.y_test = self.test_set[self.target]

        logger.info("✅ Feature engineering completed.")

    def train(self):
        """Train the model"""
        logger.info("Training the model...")

        # Define the preprocessor step
        cat_pipeline = Pipeline(
            [
                (
                    "cat_imputer",
                    SimpleImputer(strategy="most_frequent", missing_values=None),
                ),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        num_pipeline = Pipeline(
            [
                ("num_imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        preprocessor = ColumnTransformer(
            [
                ("num", num_pipeline, self.num_features),
                ("cat", cat_pipeline, self.cat_features),
            ],
            remainder="passthrough",
        )

        # Define the final pipeline
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", LGBMRegressor(**self.parameters)),
            ]
        )

        # Train model and log metrics, parameters, and model
        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id
            pipeline.fit(self.X_train, self.y_train)
            y_pred = pipeline.predict(self.X_test)

            # Compute evaluation metrics
            mse = mean_squared_error(self.y_test, y_pred)
            mae = mean_absolute_error(self.y_test, y_pred)

            # Log metrics with logger
            logger.info(f"Mean Squared Error: {mse}")
            logger.info(f"Mean Absolute Error: {mae}")

            # Log metrics in mlflow
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("mae", mae)

            # Log parameters in mlflow
            mlflow.log_param("model_type", "LGBMRegressor with preprocessing step")
            mlflow.log_params(self.parameters)

            # Infer signature
            signature = infer_signature(self.X_train, y_pred)

            # Log model with feature engineering client
            self.fe.log_model(
                model=pipeline,
                flavor=mlflow.sklearn,
                artifact_path="lightgbm-pipeline-model-fe",
                training_set=self.training_set_spec,
                signature=signature,
            )

    def register_model(self):
        """Register the model in the model registry"""
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/lightgbm-pipeline-model-fe",
            name=f"{self.catalog_name}.{self.ml_asset_schema}.{self.config.model.MODEL_NAME}",
            tags=self.tags,
        )

        # Get the latest version of the model (an integer)
        latest_version = registered_model.version

        client = MlflowClient()
        # Set alias for the model version
        client.set_registered_model_alias(
            name=f"{self.catalog_name}.{self.ml_asset_schema}.{self.config.model.MODEL_NAME}",
            alias="latest-model",
            version=latest_version,
        )

    def load_latest_model_and_predict(self, X):
        """Load the latest model and make predictions"""
        # Load the latest model version from MLFlow using Feature Engineering client
        model_uri = f"models:/{self.catalog_name}.{self.ml_asset_schema}.{self.config.model.MODEL_NAME}@latest-model"

        # Make predictions
        predictions = self.fe.score_batch(model_uri=model_uri, df=X)
        return predictions
