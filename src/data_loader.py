"""
Data loading utilities for the fraud detection system.
Handles loading and basic validation of the credit card transaction dataset.
"""

import os
import pandas as pd
import numpy as np
from typing import Optional
import logging
from .logging_config import LOGGING_CONFIG
import logging.config

# Configure logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

def validate_data_quality(df: pd.DataFrame) -> None:
    """
    Validate data quality and log any issues found.

    Args:
        df: DataFrame to validate
    """
    if df.isnull().any().any():
        null_counts = df.isnull().sum()
        null_cols = null_counts[null_counts > 0]
        logger.warning(f"Dataset contains missing values in columns: {null_cols.to_dict()}")

    if df['Amount'].min() < 0:
        neg_count = (df['Amount'] < 0).sum()
        logger.warning(f"Dataset contains {neg_count} negative amounts")

    # Check for extreme values
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers = (z_scores > 3).sum()
            if outliers > 0:
                logger.info(f"Column {col} has {outliers} potential outliers (|z-score| > 3)")

def load_dataset(data_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the credit card transaction dataset from the specified path or default location.

    Args:
        data_path (str, optional): Custom path to the dataset. If None, uses default path.

    Returns:
        pd.DataFrame: Loaded dataset with transaction data.

    Raises:
        FileNotFoundError: If the dataset file is not found.
        ValueError: If the loaded dataset doesn't match expected format.
    """
    try:
        if data_path is None:
            data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'creditcard.csv')
            logger.info(f"Using default data path: {data_path}")
        else:
            logger.info(f"Loading data from: {data_path}")

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at: {data_path}")

        df = pd.read_csv(data_path)
        logger.info(f"Loaded dataset with shape: {df.shape}")

        # Validate dataset structure
        required_columns = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Dataset missing required columns: {', '.join(missing_columns)}")

        # Validate data quality
        validate_data_quality(df)

        return df

    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        df = load_dataset()
        logger.info("\nDataset Summary:")
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info("\nSample statistics:")
        logger.info(f"Fraud cases: {(df['Class'] == 1).sum() if 'Class' in df.columns else 'N/A'}")
        logger.info(f"Average transaction amount: ${df['Amount'].mean():.2f}")
    except Exception as e:
        logger.error(f"Error in data loading script: {str(e)}")
