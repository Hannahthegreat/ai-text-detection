from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
import joblib
import pickle
from pathlib import Path
import time

def train_logistic_regression(X_train, y_train, class_weight=None, max_iter=1000):
    """
    Train Logistic Regression model.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    class_weight : dict or 'balanced', optional
        Class weights for handling imbalance
    max_iter : int
        Maximum iterations for convergence
    
    Returns:
    --------
    tuple
        (model, training_time)
    """
    print("\nTraining Logistic Regression...")
    start_time = time.time()
    
    model = LogisticRegression(
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    return model, training_time


def train_naive_bayes(X_train, y_train):
    """
    Train Multinomial Naive Bayes model.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    
    Returns:
    --------
    tuple
        (model, training_time)
    """
    print("\nTraining Naive Bayes...")
    start_time = time.time()
    
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    return model, training_time


def train_random_forest(X_train, y_train, n_estimators=100, 
                       max_depth=None, class_weight=None):
    """
    Train Random Forest model.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    n_estimators : int
        Number of trees
    max_depth : int, optional
        Maximum depth of trees
    class_weight : dict or 'balanced', optional
        Class weights
    
    Returns:
    --------
    tuple
        (model, training_time)
    """
    print("\nTraining Random Forest...")
    start_time = time.time()
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    return model, training_time


def train_xgboost(X_train, y_train, scale_pos_weight=None, 
                 n_estimators=100, max_depth=6):
    """
    Train XGBoost model.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    scale_pos_weight : float, optional
        Weight for positive class
    n_estimators : int
        Number of boosting rounds
    max_depth : int
        Maximum tree depth
    
    Returns:
    --------
    tuple
        (model, training_time)
    """
    print("\nTraining XGBoost...")
    start_time = time.time()
    
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    return model, training_time


def create_soft_voting_ensemble(models_dict, X_train, y_train, voting='soft'):
    """
    Create soft voting ensemble from multiple models.
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary of (name, model) pairs
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    voting : str
        'soft' or 'hard' voting
    
    Returns:
    --------
    VotingClassifier
        Trained ensemble model
    """
    print(f"\nCreating {voting} voting ensemble...")
    print(f"Models: {list(models_dict.keys())}")
    
    estimators = [(name, model) for name, model in models_dict.items()]
    
    ensemble = VotingClassifier(
        estimators=estimators,
        voting=voting
    )
    
    start_time = time.time()
    ensemble.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"Ensemble training completed in {training_time:.2f} seconds")
    
    return ensemble


def save_model(model, vectorizer, filepath='models/saved_models/model.joblib', 
               metadata=None):
    """
    Save trained model and vectorizer.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    vectorizer : TfidfVectorizer
        Fitted vectorizer
    filepath : str
        Path to save the model
    metadata : dict, optional
        Additional metadata to save
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    model_data = {
        'model': model,
        'vectorizer': vectorizer,
        'metadata': metadata or {}
    }
    
    joblib.dump(model_data, filepath)
    print(f"\nModel saved to: {filepath}")


def load_model(filepath='models/saved_models/model.joblib'):
    """
    Load trained model and vectorizer.
    
    Parameters:
    -----------
    filepath : str
        Path to the saved model
    
    Returns:
    --------
    dict
        Dictionary with 'model', 'vectorizer', and 'metadata'
    """
    model_data = joblib.load(filepath)
    print(f"Model loaded from: {filepath}")
    
    if 'metadata' in model_data:
        print(f"Metadata: {model_data['metadata']}")
    
    return model_data


def get_feature_importance(model, vectorizer, n=20):
    """
    Get feature importance from tree-based models.
    
    Parameters:
    -----------
    model : sklearn model
        Trained Random Forest or XGBoost
    vectorizer : TfidfVectorizer
        Fitted vectorizer
    n : int
        Number of top features to return
    
    Returns:
    --------
    pandas.DataFrame
        Feature importance dataframe
    """
    import pandas as pd
    
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have feature_importances_ attribute")
        return None
    
    feature_names = vectorizer.get_feature_names_out()
    importances = model.feature_importances_
    
    feature_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    feature_imp = feature_imp.sort_values('importance', ascending=False)
    
    print(f"\nTop {n} Important Features:")
    print(feature_imp.head(n).to_string(index=False))
    
    return feature_imp