"""
Streamlit web application for fraud detection in financial transactions.
Provides an interface for uploading transaction data and detecting potential fraud.
"""

import streamlit as st
import pandas as pd
import os
import joblib
from src.preprocess import preprocess_data
import logging
from typing import Optional, Tuple
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define paths using absolute paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'isolation_forest.pkl')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')

def load_model_and_scaler() -> Optional[Tuple[object, object]]:
    """
    Load the trained model and scaler.
    Returns:
        tuple: (model, scaler) if successful, None otherwise
    """
    try:
        if not os.path.exists(MODEL_PATH):
            st.error("⚠️ Model not found. Please train the model first.")
            st.info("Run 'python src/model.py' to train the model.")
            return None

        if not os.path.exists(SCALER_PATH):
            st.error("⚠️ Scaler not found. Please train the model first.")
            st.info("Run 'python src/model.py' to train the model.")
            return None

        model = joblib.load(MODEL_PATH)
        st.success("✅ Model loaded successfully")
        return model

    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        st.error(f"❌ Failed to load model: {str(e)}")
        return None

def validate_input_data(df: pd.DataFrame) -> bool:
    """
    Validate the uploaded data format.
    Args:
        df: pandas DataFrame to validate
    Returns:
        bool: True if valid, False otherwise
    """
    required_cols = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
        return False

    if df.isnull().any().any():
        st.warning("⚠️ Dataset contains missing values. These rows will be excluded.")

    return True

def process_predictions(df: pd.DataFrame, scores: np.ndarray, predictions: np.ndarray) -> pd.DataFrame:
    """
    Process model predictions and add them to the DataFrame.
    
    Args:
        df: Original DataFrame
        scores: Anomaly scores from model
        predictions: Model predictions (-1 for fraud, 1 for normal)
        
    Returns:
        DataFrame with predictions and scores added
    """
    df = df.copy()
    df['Fraud_Pred'] = ['🚨 Fraud' if p == -1 else '✅ Normal' for p in predictions]
    df['Anomaly_Score'] = scores
    return df

def main():
    """Main application function."""
    # Set up Streamlit page
    st.set_page_config(
        page_title="Fraud Detection System",
        page_icon="🔍",
        layout="wide"
    )

    st.title('🔍 Fintech Fraud Detection System')
    st.write("Upload a CSV file with transaction data to detect potential fraud")

    with st.expander("ℹ️ Input Format Requirements"):
        st.write("""
        Your CSV file should contain the following columns:
        - Time: Transaction timestamp
        - Amount: Transaction amount
        - V1-V28: Transaction features
        
        The file should not contain any missing values.
        """)

    uploaded_file = st.file_uploader("Upload transaction CSV", type=['csv'])

    if uploaded_file:
        try:
            # Load and validate data
            with st.spinner("📂 Loading data..."):
                df = pd.read_csv(uploaded_file)
                st.write(f"📊 Uploaded data shape: {df.shape}")

            if not validate_input_data(df):
                st.stop()

            # Load model
            model = load_model_and_scaler()
            if model is None:
                st.stop()

            # Process data for inference
            with st.spinner("⚙️ Processing data..."):
                X, _, _ = preprocess_data(df, is_training=False)
                st.success("✅ Data preprocessing complete")

            # Make predictions
            with st.spinner("🔍 Detecting fraud..."):
                predictions = model.predict(X)
                scores = -model.decision_function(X)
                df = process_predictions(df, scores, predictions)
                st.success("✅ Fraud detection complete")

            # Show results
            st.subheader("📊 Results")
            fraud_count = (df['Fraud_Pred'] == '🚨 Fraud').sum()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Transactions", len(df))
            with col2:
                st.metric("Potential Fraud", fraud_count)
            with col3:
                st.metric("Fraud Rate", f"{(fraud_count/len(df))*100:.2f}%")

            # Show the dataframe with predictions
            st.dataframe(
                df.style.background_gradient(
                    subset=['Anomaly_Score'],
                    cmap='RdYlGn_r'
                )
            )

            # Download option
            st.download_button(
                '📥 Download Results CSV',
                df.to_csv(index=False),
                'fraud_detection_results.csv',
                'text/csv',
                key='download-csv'
            )

        except Exception as e:
            logging.error(f"Error processing file: {str(e)}")
            st.error(f"❌ An error occurred: {str(e)}")
            st.write("Please check your input file format and try again.")

if __name__ == "__main__":
    main()
