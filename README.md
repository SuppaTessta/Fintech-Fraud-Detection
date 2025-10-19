# Fintech Fraud Detection System

A machine learning-based system for detecting fraudulent financial transactions using Isolation Forest algorithm, featuring real-time inference through a Streamlit web interface.

## 📊 Dataset

This project uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/) from Kaggle. The dataset contains transactions made by credit cards in September 2013 by European cardholders. It presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions.

### Dataset Features:
- Features V1-V28 are principal components obtained with PCA transformation
- 'Time' represents seconds elapsed between each transaction and the first transaction
- 'Amount' is the transaction amount
- 'Class' is the response variable (1 for fraud, 0 for normal)

### Dataset Citation
The dataset has been collected and analyzed during a research collaboration of Worldline and the Machine Learning Group (http://mlg.ulb.ac.be) of ULB (Université Libre de Bruxelles) on big data mining and fraud detection.

## 🌟 Features

- Real-time fraud detection using Isolation Forest algorithm
- Interactive web interface with Streamlit
- Advanced feature engineering for better detection
- Robust data preprocessing pipeline
- Detailed model evaluation and visualization
- Support for large-scale transaction data
- Configurable model parameters and preprocessing steps
- Comprehensive logging system

## 🛠 Technology Stack

- **Python 3.8+**
- **Core Libraries:**
  - scikit-learn (Isolation Forest)
  - pandas & numpy (Data processing)
  - Streamlit (Web interface)
- **Visualization:**
  - matplotlib
  - seaborn
- **Data Handling:**
  - joblib (Model persistence)
  - PyYAML (Configuration)

## 📁 Project Structure

```
fintech-fraud-detection/
│
├── app.py              # Main Streamlit web application
├── requirements.txt    # Project dependencies
├── data/              # Data directory
│   └── creditcard.csv # Training/test data (not included in repo)
├── models/            # Saved model artifacts
│   ├── isolation_forest.pkl
│   └── scaler.pkl
└── src/              # Source code
    ├── __init__.py
    ├── config.py     # Configuration management
    ├── data_loader.py    # Data loading utilities
    ├── model.py      # Model training and evaluation
    ├── preprocess.py # Data preprocessing pipeline
    └── logging_config.py # Logging configuration
```

## 🚀 Quick Start

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd fintech-fraud-detection
   ```

2. **Set up Python environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   python -m pip install -r requirements.txt
   ```

3. **Prepare your data:**
   - Place your transaction data CSV in the `data/` directory
   - Required columns: 'Time', 'Amount', 'V1' through 'V28'
   - For training: include 'Class' column (0 for normal, 1 for fraud)

4. **Train the model:**
   ```bash
   python -m src.model
   ```

5. **Run the web application:**
   ```bash
   streamlit run app.py
   ```

## 💻 Usage

### Training Mode

To train a new model:
1. Ensure your training data is in `data/creditcard.csv`
2. Run the training script:
   ```bash
   python -m src.model
   ```
3. The script will:
   - Load and preprocess the data
   - Train the Isolation Forest model
   - Generate performance metrics
   - Save the model and scaler to `models/`

### Inference Mode

1. Start the web application:
   ```bash
   streamlit run app.py
   ```
2. Upload a CSV file with transaction data
3. View real-time fraud detection results
4. Download the results as CSV

## 🔧 Configuration

The system is configurable through `src/config.py`:

```python
DEFAULT_CONFIG = {
    'model': {
        'contamination': 0.0017,  # Expected fraud ratio
        'random_state': 42,
        'n_estimators': 100,
        'max_samples': 'auto',
        'n_jobs': -1
    },
    'preprocessing': {
        'undersample_ratio': 0.5,  # For handling imbalanced data
        'random_state': 42
    }
    # ...other settings
}
```

## 📊 Feature Engineering

The system performs advanced feature engineering:

- **Time-based features:**
  - Hour of day
  - Night transaction flag
- **Amount-based features:**
  - Log-transformed amount
  - Large transaction flag
- **Standardization:**
  - All numerical features are standardized
  - Scaling parameters are saved for inference

## 📝 Logging

Comprehensive logging is implemented throughout:
- Training progress and metrics
- Data preprocessing steps
- Model performance metrics
- Runtime errors and warnings

Logs are stored in the `logs/` directory.

## 🔍 Model Evaluation

The system provides detailed model evaluation:
- Confusion matrix
- Precision-Recall curves
- Classification report
- Custom metrics:
  - Fraud Detection Rate
  - False Positive Rate
  - PR-AUC Score

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📫 Support

For support, please open an issue in the GitHub repository.
