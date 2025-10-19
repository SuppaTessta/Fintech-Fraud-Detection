"""Data preprocessing module for the fraud detection system.

This module handles all data preprocessing steps including:
- Feature scaling
- Data validation
- Training/inference mode handling
- Feature engineering
"""
from typing import Tuple, Optional, Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import joblib
import logging
from .logging_config import LOGGING_CONFIG
from .config import get_config
import logging.config

# Configure logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional features from existing data.

    Args:
        df: Input DataFrame with Time and Amount columns

    Returns:
        DataFrame with additional engineered features
    """
    df = df.copy()

    # Time-based features
    df['Hour'] = (df['Time'] / 3600) % 24
    df['IsNight'] = ((df['Hour'] >= 22) | (df['Hour'] <= 5)).astype(int)

    # Amount-based features
    df['AmountLog'] = np.log1p(df['Amount'])  # Log transform for better distribution
    df['IsLargeTransaction'] = (df['Amount'] > df['Amount'].quantile(0.95)).astype(int)

    return df

def preprocess_data(
    df: pd.DataFrame,
    is_training: bool = True
) -> Tuple[pd.DataFrame, Optional[pd.Series], Optional[StandardScaler]]:
    """
    Preprocess data for training or inference.

    Args:
        df: Input DataFrame with Time, Amount columns (and Class for training)
        is_training: If True, fit scaler and return labels; if False, use saved scaler

    Returns:
        Tuple containing:
        - X: Preprocessed features DataFrame
        - y: Labels Series (None for inference)
        - scaler: StandardScaler object (None for inference)

    Raises:
        ValueError: If required columns are missing
        FileNotFoundError: If scaler file is not found in inference mode
    """
    try:
        logger.info(f"Starting preprocessing in {'training' if is_training else 'inference'} mode")
        config = get_config()

        # Create a copy to avoid modifying the original
        df_copy = df.copy()

        # Validate input data
        required_cols = ['Time', 'Amount']
        if is_training:
            required_cols.append('Class')

        missing_cols = [col for col in required_cols if col not in df_copy.columns]
        if missing_cols:
            error_msg = f"Missing required columns: {', '.join(missing_cols)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Check for V1-V28 columns
        v_cols = [f'V{i}' for i in range(1, 29)]
        missing_v_cols = [col for col in v_cols if col not in df_copy.columns]
        if missing_v_cols:
            error_msg = f"Missing feature columns: {', '.join(missing_v_cols)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Handle missing values
        if df_copy.isnull().any().any():
            null_counts = df_copy.isnull().sum()
            logger.warning(f"Found missing values:\n{null_counts[null_counts > 0]}")
            df_copy.dropna(inplace=True)
            logger.info(f"Dropped {len(df) - len(df_copy)} rows with missing values")

        # Engineer additional features
        df_copy = engineer_features(df_copy)
        logger.info("Added engineered features")

        # Scale features
        if is_training:
            logger.info("Fitting new scaler")
            scaler = StandardScaler()
            features_to_scale = ['Amount', 'Time', 'AmountLog', 'Hour']
            scaled_values = scaler.fit_transform(df_copy[features_to_scale])

            # Save the scaler
            models_dir = config['paths']['models_dir']
            os.makedirs(models_dir, exist_ok=True)
            scaler_path = os.path.join(models_dir, 'scaler.pkl')
            joblib.dump(scaler, scaler_path)
            logger.info(f"Saved scaler to: {scaler_path}")
        else:
            logger.info("Loading pre-fitted scaler")
            scaler_path = os.path.join(config['paths']['models_dir'], 'scaler.pkl')
            if not os.path.exists(scaler_path):
                error_msg = f"Scaler not found at {scaler_path}. Please train the model first."
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)

            scaler = joblib.load(scaler_path)
            features_to_scale = ['Amount', 'Time', 'AmountLog', 'Hour']
            scaled_values = scaler.transform(df_copy[features_to_scale])

        # Add scaled features back to DataFrame
        for i, col in enumerate(features_to_scale):
            df_copy[f'{col}_scaled'] = scaled_values[:, i]

        # Drop original unscaled features
        df_copy.drop(features_to_scale, axis=1, inplace=True)

        # Prepare final feature set
        if is_training and 'Class' in df_copy.columns:
            logger.info("Preparing training data with undersampling")
            # Get configuration for undersampling
            undersample_ratio = config['preprocessing']['undersample_ratio']
            random_state = config['preprocessing']['random_state']

            normal = df_copy[df_copy['Class'] == 0].sample(
                frac=undersample_ratio,
                random_state=random_state
            )
            fraud = df_copy[df_copy['Class'] == 1]
            df_balanced = pd.concat([normal, fraud])

            X = df_balanced.drop('Class', axis=1)
            y = df_balanced['Class']
            logger.info(f"Final training set shape: {X.shape}")
        else:
            if 'Class' in df_copy.columns:
                y = df_copy['Class'].copy()
                X = df_copy.drop('Class', axis=1)
            else:
                X = df_copy
                y = None
            logger.info(f"Final feature set shape: {X.shape}")

        return X, y, scaler

    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        from .data_loader import load_dataset

        # Test preprocessing pipeline
        df = load_dataset()
        X, y, scaler = preprocess_data(df, is_training=True)

        logger.info("\nPreprocessing Summary:")
        logger.info(f"Input shape: {df.shape}")
        logger.info(f"Output shape: {X.shape}")
        logger.info(f"Features: {list(X.columns)}")
        if y is not None:
            logger.info(f"Class distribution:\n{y.value_counts(normalize=True)}")

    except Exception as e:
        logger.error(f"Preprocessing test failed: {str(e)}")
        raise
