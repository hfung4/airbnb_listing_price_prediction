import mlflow
from databricks.connect import DatabricksSession

from airbnb_listing.config import Tags, config
from airbnb_listing.models.feature_lookup_model import FeatureLookUpModel

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

spark = DatabricksSession.builder.getOrCreate()

# raw tags
tags_dict = {"git_sha": "abcd12345", "branch": "week2"}
# validated tags
tags = Tags(**tags_dict)

# Initialize the FeatureLookUpModel
fe_model = FeatureLookUpModel(config=config, tags=tags, spark=spark)

# Create the feature table
fe_model.create_feature_table()

# Define the `days_since_last_review` feature function
fe_model.create_feature_function()

# Load silver data
fe_model.load_silver_data()

# Perform feature engineering and create training set
fe_model.feature_engineering()

# Train the model
fe_model.train()

# Register the model
fe_model.register_model()
