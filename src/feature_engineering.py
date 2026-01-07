from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from pathlib import Path


def create_tfidf_features(X_train, X_val, X_test, max_features=5000, 
                          ngram_range=(1, 2), min_df=5, max_df=0.8):
    """
    Create TF-IDF features for text data.
    
    Parameters:
    -----------
    X_train, X_val, X_test : array-like
        Text data for each split
    max_features : int
        Maximum number of features to extract
    ngram_range : tuple
        Range of n-grams (unigrams, bigrams, etc.)
    min_df : int
        Minimum document frequency
    max_df : float
        Maximum document frequency (as proportion)
    
    Returns:
    --------
    tuple
        (X_train_tfidf, X_val_tfidf, X_test_tfidf, vectorizer)
    """
    print(f"\nCreating TF-IDF features...")
    print(f"Parameters: max_features={max_features}, ngram_range={ngram_range}")
    print(f"            min_df={min_df}, max_df={max_df}")
    
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df
    )
    
    # Fit on training data only
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)
    X_test_tfidf = tfidf.transform(X_test)
    
    print(f"TF-IDF shape: {X_train_tfidf.shape}")
    print(f"Vocabulary size: {len(tfidf.vocabulary_)}")
    
    return X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf


def save_tfidf_features(X_train, X_val, X_test, vectorizer, 
                        filepath='../data/features/tfidf_features_medium.pkl'):
    """
    Save TF-IDF features and vectorizer to pickle file.
    
    Parameters:
    -----------
    X_train, X_val, X_test : sparse matrices
        TF-IDF transformed features
    vectorizer : TfidfVectorizer
        Fitted vectorizer object
    filepath : str
        Path to save the pickle file
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    tfidf_data = {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'vectorizer': vectorizer
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(tfidf_data, f)
    
    print(f"\nTF-IDF features saved to: {filepath}")


def load_tfidf_features(filepath='../data/features/tfidf_features_medium.pkl'):
    """
    Load TF-IDF features and vectorizer from pickle file.
    
    Parameters:
    -----------
    filepath : str
        Path to the pickle file
    
    Returns:
    --------
    dict
        Dictionary with keys: X_train, X_val, X_test, vectorizer
    """
    with open(filepath, 'rb') as f:
        tfidf_data = pickle.load(f)
    
    print(f"TF-IDF features loaded from: {filepath}")
    print(f"Training set shape: {tfidf_data['X_train'].shape}")
    print(f"Validation set shape: {tfidf_data['X_val'].shape}")
    print(f"Test set shape: {tfidf_data['X_test'].shape}")
    
    return tfidf_data


def get_top_tfidf_features(vectorizer, n=20):
    """
    Get top N features by TF-IDF score.
    
    Parameters:
    -----------
    vectorizer : TfidfVectorizer
        Fitted vectorizer
    n : int
        Number of top features to return
    
    Returns:
    --------
    list
        Top N feature names
    """
    feature_names = vectorizer.get_feature_names_out()
    print(f"\nTop {n} TF-IDF features:")
    for i, feature in enumerate(feature_names[:n], 1):
        print(f"{i}. {feature}")
    
    return list(feature_names[:n])