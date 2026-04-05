import pandas as pd
import os
import json


MEDIANS_PATH = "model/training_medians.json"

def preprocess_training_data(df, is_training=True, filename="processed_train.csv"):
    """
    DEVELOPER SIDE: Cleans data. If is_training=True, it learns and saves the medians.
    If is_training=False, it loads the medians to prevent leakage.
    """
    df = df.copy()
    
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
        
    os.makedirs(os.path.dirname(MEDIANS_PATH), exist_ok=True)
        
    if is_training:
        # 1 Calculating medians ONLY from the training split
        income_median = float(df['MonthlyIncome'].median())
        dep_median = float(df['NumberOfDependents'].median())
        
        # 2 Saving them to a file for the test set and Streamlit to use
        with open(MEDIANS_PATH, 'w') as f:
            json.dump({'MonthlyIncome': income_median, 'NumberOfDependents': dep_median}, f)
    else:
        # 3 Loading the saved medians for the test set
        with open(MEDIANS_PATH, 'r') as f:
            medians = json.load(f)
        income_median = medians['MonthlyIncome']
        dep_median = medians['NumberOfDependents']

    # Applying the medians
    df['MonthlyIncome'] = df['MonthlyIncome'].fillna(income_median)
    df['NumberOfDependents'] = df['NumberOfDependents'].fillna(dep_median)
    
    # Feature engineering
    df['total_late_payments'] = (
        df['NumberOfTimes90DaysLate'] + 
        df['NumberOfTime30-59DaysPastDueNotWorse'] + 
        df['NumberOfTime60-89DaysPastDueNotWorse']
    )
    df['debt_income_ratio'] = df['DebtRatio'] / (df['MonthlyIncome'] + 1)
    df['credit_pressure'] = df['RevolvingUtilizationOfUnsecuredLines'] * df['DebtRatio']
    
    # Saving the cleaned data to a new CSV file in the processed folder
    output_dir = "data/processed/training_data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{filename}"
    
    df.to_csv(output_path, index=False)
    print(f"Success! Cleaned data saved to {output_path}")
    
    return df


def preprocess_new_data(df, filename="processed_uploaded.csv"):
    """
    CLIENT SIDE: Streamlit uses this. It loads the saved medians from the training 
    phase to ensure zero leakage when evaluating new customers.
    """
    df = df.copy()
    
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
        
    # Loading training median
    try:
        with open(MEDIANS_PATH, 'r') as f:
            medians = json.load(f)
        income_median = medians['MonthlyIncome']
        dep_median = medians['NumberOfDependents']
    except FileNotFoundError:
        # Fallback just in case someone deletes the file
        income_median = 0
        dep_median = 0
        
    df['MonthlyIncome'] = df['MonthlyIncome'].fillna(income_median)
    df['NumberOfDependents'] = df['NumberOfDependents'].fillna(dep_median)
    
    # Feature engineering
    df['total_late_payments'] = (
        df['NumberOfTimes90DaysLate'] + 
        df['NumberOfTime30-59DaysPastDueNotWorse'] + 
        df['NumberOfTime60-89DaysPastDueNotWorse']
    )
    df['debt_income_ratio'] = df['DebtRatio'] / (df['MonthlyIncome'] + 1)
    df['credit_pressure'] = df['RevolvingUtilizationOfUnsecuredLines'] * df['DebtRatio']
    
    # Saving the cleaned data to a new CSV file in the processed folder
    output_dir = "data/processed/uploaded_data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{filename}"
    
    df.to_csv(output_path, index=False)
    
    return df