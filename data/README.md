# Data Directory

Place your transaction data files here.

## Required File Format

The main data file should be named `creditcard.csv` and contain the following columns:
- `Time`: Transaction timestamp
- `Amount`: Transaction amount
- `V1` through `V28`: Transaction features
- `Class`: (For training data only) 0 for normal transactions, 1 for fraud

## Data Security Note

This directory is configured to ignore CSV files in git (see .gitignore) to prevent accidental commit of sensitive data.
Please ensure proper data handling and security practices when working with financial transaction data.
