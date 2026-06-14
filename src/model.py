import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import *

# =========================
# 1. Data preparation
# =========================

def separate_features_target(df, target_column):
    """
    Separate features (X) and target variable (y).

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataset.
    target_column : str
        Name of the target column.

    Returns:
    --------
    X : pandas.DataFrame
        Feature matrix.
    y : pandas.Series
        Target variable.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y


def stratified_split(X, y, test_size=0.2, random_state=42):
    """
    Perform stratified train-test split to preserve class distribution.

    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix.
    y : pandas.Series
        Target variable.
    test_size : float, default=0.2
        Proportion of test data.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns:
    --------
    X_train, X_test, y_train, y_test
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,          # Preserves class distribution
        random_state=random_state
    )

# =========================
# 2.Build Baseline Model
# =========================

def train_logistic_regression(X_train, y_train, random_state=42):
    """
    Train a Logistic Regression model as a baseline.
    """
    model = LogisticRegression(
        max_iter=1000,
        solver='lbfgs',
        random_state=random_state
    )

    model.fit(X_train, y_train)
    return model

def get_predictions(model, X_test):
    """
    Get predicted labels and probabilities.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # probability of positive class

    return y_pred, y_proba

def evaluate_classification_model(y_true, y_pred, y_proba):
    """
    Evaluate model using:
    - AUC-PR (Average Precision Score)
    - F1 Score
    - Confusion Matrix
    """

    auc_pr = average_precision_score(y_true, y_proba)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print("\n================ MODEL EVALUATION ================\n")

    print(f"AUC-PR Score (Average Precision): {auc_pr:.4f}")
    print(f"F1 Score                       : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    return {
        "auc_pr": auc_pr,
        "f1_score": f1,
        "confusion_matrix": cm
    }

def run_logistic_regression_pipeline(X_train, X_test, y_train, y_test):
    """
    Complete pipeline: train → predict → evaluate
    """

    # Train model
    model = train_logistic_regression(X_train, y_train)

    # Predictions
    y_pred, y_proba = get_predictions(model, X_test)

    # Evaluation
    results = evaluate_classification_model(y_test, y_pred, y_proba)

    return model, results

# =========================
# 3. Build Ensemble Model
# =========================

def get_param_grid(model_name):
    """
    Hyperparameter search space.
    """

    param_grids = {

        "random_forest": {
            "n_estimators": [100, 200],
            "max_depth": [5, 10, None]
        },

        "xgboost": {
            "n_estimators": [100, 200],
            "max_depth": [3, 6]
        },

        "lightgbm": {
            "n_estimators": [100, 200],
            "max_depth": [5, 10]
        }
    }

    return param_grids[model_name]

def get_param_grid(model_name):
    """
    Hyperparameter search space.
    """

    param_grids = {

        "random_forest": {
            "n_estimators": [100, 200],
            "max_depth": [5, 10, None]
        },

        "xgboost": {
            "n_estimators": [100, 200],
            "max_depth": [3, 6]
        },

        "lightgbm": {
            "n_estimators": [100, 200],
            "max_depth": [5, 10]
        }
    }

    return param_grids[model_name]

def tune_model(model, param_grid, X_train, y_train):
    """
    Perform Grid Search tuning.
    """

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="average_precision",
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print("\nBest Parameters:")
    print(grid_search.best_params_)

    return grid_search.best_estimator

def generate_predictions(model, X_test):
    """
    Generate predictions and probabilities.
    """

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return y_pred, y_proba

def evaluate_model(y_test, y_pred, y_proba, model_name):
    """
    Evaluate model performance.
    """

    auc_pr = average_precision_score(y_test, y_proba)

    f1 = f1_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"{model_name.upper()} RESULTS")
    print(f"{'='*50}")

    print(f"AUC-PR Score : {auc_pr:.4f}")
    print(f"F1 Score     : {f1:.4f}")

    print("\nConfusion Matrix")
    print(cm)

    print("\nClassification Report")
    print(classification_report(y_test, y_pred))

    return {
        "auc_pr": auc_pr,
        "f1_score": f1,
        "confusion_matrix": cm
    }

def train_tune_evaluate(
    model_name,
    X_train,
    y_train,
    X_test,
    y_test
):
    """
    Complete pipeline:
    Create Model
    → Tune Model
    → Predict
    → Evaluate
    """

    # Create model
    model = create_model(model_name)

    # Get hyperparameters
    param_grid = get_param_grid(model_name)

    # Tune model
    best_model = tune_model(
        model,
        param_grid,
        X_train,
        y_train
    )

    # Predictions
    y_pred, y_proba = generate_predictions(
        best_model,
        X_test
    )

    # Evaluation
    results = evaluate_model(
        y_test,
        y_pred,
        y_proba,
        model_name
    )

    return best_model, results

def compare_models(
    X_train,
    y_train,
    X_test,
    y_test
):
    """
    Train and compare all ensemble models.
    """

    models = [
        "random_forest",
        "xgboost",
        "lightgbm"
    ]

    results = {}

    for model_name in models:

        print(f"\nTraining {model_name}...")

        model, metrics = train_tune_evaluate(
            model_name,
            X_train,
            y_train,
            X_test,
            y_test
        )

        results[model_name] = metrics

    return results


# =========================
# 4. Cross-Validation 
# =========================

def build_model(n_estimators=100, max_depth=None, random_state=42):
    """
    Create a model instance.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )

def train_evaluate_fold(model, X_train, y_train, X_val, y_val):
    """
    Train model on fold and evaluate.
    """

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    auc_pr = average_precision_score(y_val, y_proba)
    f1 = f1_score(y_val, y_pred)
    cm = confusion_matrix(y_val, y_pred)

    return auc_pr, f1, cm

def stratified_kfold_evaluation(
    X, y,
    n_splits=5,
    n_estimators=100,
    max_depth=None,
    random_state=42
):
    """
    Perform Stratified K-Fold CV and compute metrics.
    """

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    auc_pr_scores = []
    f1_scores = []
    confusion_matrices = []

    fold = 1

    for train_idx, val_idx in skf.split(X, y):

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = build_model(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )

        auc_pr, f1, cm = train_evaluate_fold(
            model, X_train, y_train, X_val, y_val
        )

        auc_pr_scores.append(auc_pr)
        f1_scores.append(f1)
        confusion_matrices.append(cm)

        print(f"Fold {fold} → AUC-PR: {auc_pr:.4f}, F1: {f1:.4f}")
        fold += 1

    return auc_pr_scores, f1_scores, confusion_matrices

def summarize_cv_results(auc_pr_scores, f1_scores):
    """
    Compute mean and std for CV metrics.
    """

    print("\n========== CROSS-VALIDATION SUMMARY ==========\n")

    print(f"AUC-PR → Mean: {np.mean(auc_pr_scores):.4f}, Std: {np.std(auc_pr_scores):.4f}")
    print(f"F1     → Mean: {np.mean(f1_scores):.4f}, Std: {np.std(f1_scores):.4f}")

def run_stratified_kfold_pipeline(X, y, n_splits=5):
    """
    Full Stratified K-Fold evaluation pipeline.
    """

    auc_pr_scores, f1_scores, cms = stratified_kfold_evaluation(
        X, y,
        n_splits=n_splits
    )

    summarize_cv_results(auc_pr_scores, f1_scores)

    return {
        "auc_pr_scores": auc_pr_scores,
        "f1_scores": f1_scores,
        "confusion_matrices": cms
    }