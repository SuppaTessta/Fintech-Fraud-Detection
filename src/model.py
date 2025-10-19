"""
Model training and evaluation module for the fraud detection system.

This module handles:
- Model training with Isolation Forest
- Model evaluation and visualization
- Model persistence
"""

from typing import Optional, Tuple, Dict, Any
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
import logging
from .logging_config import LOGGING_CONFIG
from .config import get_config
import logging.config

# Configure logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def train_model(
    X: pd.DataFrame,
    contamination: Optional[float] = None
) -> IsolationForest:
    """
    Train an Isolation Forest model for fraud detection.

    Args:
        X: Feature DataFrame
        contamination: Expected proportion of outliers in the dataset

    Returns:
        Trained IsolationForest model

    Raises:
        ValueError: If input data is invalid
        IOError: If model saving fails
    """
    try:
        # Get configuration
        config = get_config()
        if contamination is None:
            contamination = config['model']['contamination']

        logger.info(f"Training Isolation Forest model with contamination={contamination}")

        if X.empty:
            raise ValueError("Empty training data provided")

        # Create models directory if it doesn't exist
        models_dir = config['paths']['models_dir']
        os.makedirs(models_dir, exist_ok=True)

        # Initialize model with configuration
        model = IsolationForest(
            contamination=contamination,
            random_state=config['model']['random_state'],
            n_estimators=config['model']['n_estimators'],
            max_samples=config['model']['max_samples'],
            n_jobs=config['model']['n_jobs']
        )

        # Fit model
        logger.info(f"Starting model training with {len(X)} samples")
        model.fit(X)
        logger.info("Model training completed")

        # Save model
        model_path = os.path.join(models_dir, 'isolation_forest.pkl')
        joblib.dump(model, model_path)
        logger.info(f"Saved model to: {model_path}")

        return model

    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise


def evaluate_model(
    model: IsolationForest,
    X: pd.DataFrame,
    y: pd.Series,
    save_plots: bool = True
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Evaluate the trained model and generate performance metrics.

    Args:
        model: Trained IsolationForest model
        X: Feature DataFrame
        y: True labels
        save_plots: Whether to save visualization plots

    Returns:
        Tuple of (performance_metrics, anomaly_scores)

    Raises:
        ValueError: If inputs are invalid
    """
    try:
        logger.info("Starting model evaluation")
        config = get_config()

        if X.shape[0] != len(y):
            raise ValueError("Feature and label dimensions don't match")

        # Get predictions
        anomalies = model.predict(X)  # -1 anomaly, 1 normal
        y_pred = [1 if x == -1 else 0 for x in anomalies]  # Map to fraud=1

        # Generate report
        report = classification_report(y, y_pred, output_dict=True)
        logger.info(f"Classification Report:\n{classification_report(y, y_pred)}")

        # Calculate anomaly scores
        scores = -model.decision_function(X)  # Higher score = more anomalous

        if save_plots:
            plots_dir = config['paths']['plots_dir']
            os.makedirs(plots_dir, exist_ok=True)

            # Confusion Matrix
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Fraud Detection Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.savefig(os.path.join(plots_dir, 'confusion_matrix.png'))
            plt.close()

            # Precision-Recall curve
            plt.figure(figsize=(8, 6))
            precision, recall, _ = precision_recall_curve(y, scores)
            plt.plot(recall, precision)
            plt.title('Precision-Recall Curve')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.grid(True)
            plt.savefig(os.path.join(plots_dir, 'precision_recall.png'))
            plt.close()

            logger.info(f"Saved evaluation plots to: {plots_dir}")

        # Calculate additional metrics
        metrics = {
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'auc_pr': np.trapz(precision, recall),  # Area under PR curve
            'fraud_detection_rate': report['1']['recall'],
            'false_positive_rate': cm[0, 1] / (cm[0, 0] + cm[0, 1])
        }

        logger.info("Model evaluation completed successfully")
        return metrics, scores

    except Exception as e:
        logger.error(f"Error during model evaluation: {str(e)}")
        raise


def load_model() -> Optional[IsolationForest]:
    """
    Load a trained model from disk.

    Returns:
        Loaded model or None if not found

    Raises:
        IOError: If model loading fails
    """
    try:
        config = get_config()
        model_path = os.path.join(config['paths']['models_dir'], 'isolation_forest.pkl')

        if not os.path.exists(model_path):
            logger.warning(f"No model found at: {model_path}")
            return None

        model = joblib.load(model_path)
        logger.info("Model loaded successfully")
        return model

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        from .preprocess import preprocess_data
        from .data_loader import load_dataset

        logger.info("Starting model training pipeline")

        # Load and preprocess data
        df = load_dataset()
        X, y, _ = preprocess_data(df)

        # Train model
        model = train_model(X)

        # Evaluate model
        metrics, _ = evaluate_model(model, X, y)

        # Log key metrics
        logger.info(f"Fraud Detection Rate: {metrics['fraud_detection_rate']:.2%}")
        logger.info(f"False Positive Rate: {metrics['false_positive_rate']:.2%}")
        logger.info(f"PR-AUC Score: {metrics['auc_pr']:.3f}")

        logger.info("Pipeline completed successfully")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise
