import pandas as pd
import re
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path


def load_ai_detection_dataset(dataset_name="artem9k/ai-text-detection-pile", split="train"):
    print(f"Loading dataset: {dataset_name}...")
    ds = load_dataset(dataset_name, split=split)
    df = pd.DataFrame(ds)
    print(f"Dataset loaded: {len(df):,} samples")
    return df

def convert_labels_to_numeric(df, source_col='source'):
    df['label'] = (df[source_col] == 'ai').astype(int)
    print(f"\nClass Distribution:\n{df['label'].value_counts()}")
    human_count = (df['label'] == 0).sum()
    ai_count = (df['label'] == 1).sum()
    print(f"Ratio (Human:AI) = {human_count/ai_count:.2f}:1")
    return df

def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def apply_text_cleaning(df, text_column='text'):
    print(f"Cleaning text in column '{text_column}...")
    df['cleaned_text'] = df[text_column].apply(clean_text)
    print(f"Text cleaning complete: {len(df)} samples ")
    return df

def create_stratified_sample(df, frac=0.3, random_state=42):
    print(f"\nCreating {frac*100:.0f}% stratified sample...")
    df_sample = df.sample(frac=frac, random_state=random_state)
    df_sample = df_sample.reset_index(drop=True)
    print(f"Sample size: {len(df_sample):,}")
    print(f"Class distribution:\n{df_sample['label'].value_counts()}")
    return df_sample

def split_train_val_test(X, y, test_size=0.15, val_size=0.15, random_state=42):
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted,
        random_state=random_state, stratify=y_temp
    )
    print("\nData Split Summary:")
    print(f"Training: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Validation: {len(X_val):,} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"Test: {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")
    return X_train, X_val, X_test, y_train, y_val, y_test

def save_train_test_splits(X_train, X_val, X_test, y_train, y_val, y_test,
                           filepath='data/processed/train_test_splits_medium.pkl'):
    # Save splits to pickle file
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    splits = {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
    }
    with open(filepath, 'wb') as f:
        pickle.dump(splits, f)
    print(f"\nSplits saved to: {filepath}")
    
def load_train_test_splits(filepath='data/processed/train_test_splits_medium.pkl'):
    # load splits from pickle file
    with open(filepath, 'rb') as f:
        splits = pickle.load(f)
        print(f"Splits loaded from: {filepath}")
        print(f"Train: {len(splits['X_train']):,}, Val: {len(splits['X_val']):,}, Test: {len(splits['X_test']):,}")
        return splits