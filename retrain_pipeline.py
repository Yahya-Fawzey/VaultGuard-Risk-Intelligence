import os
from sklearn.model_selection import train_test_split
from data.load_data import load_training_data
from src.preprocessing import preprocess_training_data
from src.modeling.train_model import run_training
from src.modeling.evaluate import run_evaluation

def setup_developer_backend():
    print(" STARTING DEVELOPER PIPELINE \n")
    print("="*50)

    print("STEP 1: Loading raw training data...")
    raw_df = load_training_data("data/raw/cs-training.csv")
    print(f"Loaded {len(raw_df)} rows of raw data.\n")

    print("STEP 2: Splitting Data (Preventing Data Leakage)...")
    
    train_df, test_df = train_test_split(raw_df, test_size=0.2, random_state=42)

    print("STEP 3: Cleaning training data and LEARNING medians...")
    preprocess_training_data(train_df, is_training=True, filename="processed_train.csv")

    print("STEP 4: Cleaning test data using LEARNED medians...")
    preprocess_training_data(test_df, is_training=False, filename="processed_test.csv")
    print("\n")

    print("STEP 5: Training XGBoost Model...")
    run_training()
    print("\n")

    print("STEP 6: Running final evaluation on the 20% holdout set...")
    run_evaluation()

    print("\n PIPELINE COMPLETE! The backend is fully set up.")

if __name__ == "__main__":
    setup_developer_backend()