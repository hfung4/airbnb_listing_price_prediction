import numpy as np
import pandas as pd
from databricks.connect import DatabricksSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp

from airbnb_listing.config import config
from airbnb_listing.logging import logger

spark = DatabricksSession.builder.getOrCreate()


class DataProcessor:
    def __init__(self, pandas_df: pd.DataFrame):
        self.df = pandas_df  # Store the DataFrame as self.df
        # self.config = config  # Store the configuration

    def preprocess(self) -> pd.DataFrame:
        """Preprocess the data and perform feature engineering

        Returns:
            pd.DataFrame: processed dataframe
        """
        # Drop all rows with missing values in the target column
        self.df.dropna(subset=["price"], inplace=True)

        # Convert certain float columns to Int64 (pandas nullable integer type)
        # for col in config.model.INTEGER_COLUMNS:
        #    self.df[col] = self.df[col].astype("Int64")  # Nullable integer

        # Convert the id column to a string
        self.df[config.model.ID_COLUMN] = self.df[config.model.ID_COLUMN].astype(str)

        # Log the price
        self.df["log_price"] = np.log1p(self.df["price"])

        # Drop duplicates
        self.df.drop_duplicates(inplace=True)

        # Create a is_manhattan column
        self.df["is_manhattan"] = self.df["neighbourhood_group"] == "Manhattan"

        # Cap minimum nights at 14
        self.df["minimum_nights"] = self.df["minimum_nights"].clip(upper=14)

        # elapse time since last review
        self.df["last_review"] = pd.to_datetime(self.df["last_review"], format="%Y-%m-%d", errors="coerce")
        # NOTE: days_since_last_review is now created with feature function
        # self.df["days_since_last_review"] = (
        #    datetime.now() - self.df["last_review"]
        # ).dt.days

        # Estimate for how long a house has been listed.
        # This duration is calculated by dividing the total number of reviews
        # that the house has received by the number of reviews per month.
        # Handles division by zero
        self.df["estimated_listed_months"] = np.where(
            self.df["reviews_per_month"] == 0,
            np.nan,
            self.df["number_of_reviews"] / self.df["reviews_per_month"],
        )

        # Lump rare neghbourhoods into 'Other'
        neighbourhood_percentage = self.df["neighbourhood"].value_counts(normalize=True) * 100
        self.df["neighbourhood"] = self.df["neighbourhood"].where(
            self.df["neighbourhood"].map(neighbourhood_percentage) >= config.model.THRESHOLD_NEIGHBOURHOOD,
            "Other",
        )

        # Select the columns to be used for traing
        selected_columns = (
            [config.model.ID_COLUMN]
            + config.model.SELECTED_CATEGORICAL_FEATURES
            + config.model.SELECTED_NUMERIC_FEATURES
            + config.model.SELECTED_TIMESTAMP_FEATURES
            + [config.model.TARGET]
        )
        self.df = self.df.loc[:, selected_columns]

        return self.df

    def write_processed_data(self, df: pd.DataFrame, table_name: str):
        """Write the processed data to a file

        Args:
            df (pd.DataFrame): processed dataframe
            table_name (str): three-level name of the table in Unity Catalog
        """
        # Convert the processed pandas dataFrame to a Spark DataFrame
        processed_spark = spark.createDataFrame(df).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        # Write table to Unity Catalog
        processed_spark.write.mode("append").saveAsTable(table_name)

        # Modify a Delta table property to enable Change Data Feed (CDF)
        # CDF allows tracking row-level changes (INSERT, UPDATE, DELETE) in Delta Tables.
        # With CDF enabled, you can query changes since a specific version or timestamp.
        # This is useful for incremental data processing, audting, and real-time analytics.
        spark.sql(f"ALTER TABLE {table_name} " "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")

        logger.info(f"Data written to {table_name} in Unity Catalog.")
