"""Wrapper function to feature engineering pipeline."""
import pandas as pd
import os
from utils import prepare_features, load_train_data
from config import logger


def train_model():
    final_feats_with_target_path = "data/processed_data/features/final_feats_with_target.parquet"
    final_feats_with_target = pd.read_parquet(final_feats_with_target_path)
    
    logger.info("Preparing features...")
    trusted_data = prepare_features(final_feats_with_target)
    X_train = trusted_data["X_train"]
    X_test = trusted_data["X_test"]
    y_train = pd.DataFrame(trusted_data["y_train"])
    y_test = pd.DataFrame(trusted_data["y_test"])
    
    logger.info("Saving prepared features...")
    X_train.to_parquet("data/processed_data/train/X_train.parquet")
    X_test.to_parquet("data/processed_data/train/X_test.parquet")
    y_train.to_parquet("data/processed_data/train/y_train.parquet")
    y_test.to_parquet("data/processed_data/train/y_test.parquet")

train_model()
