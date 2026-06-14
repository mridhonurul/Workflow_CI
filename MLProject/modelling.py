import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import warnings
import numpy as np
import argparse

from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

np.random.seed(42)

def load_data(data_file_path):

    df = pd.read_csv(data_file_path, sep=",")
    
    train_df, test_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42, 
        stratify=df['Outcome']
    )
    
    return train_df, test_df

def hitung_metrik(X_train, y_train, X_test, y_test, model):
    
    y_pred_train_continuous = model.predict(X_train)
    y_pred_test_continuous = model.predict(X_test)
    
    y_pred_train_class = np.where(y_pred_train_continuous >= 0.5, 1, 0)
    y_pred_test_class = np.where(y_pred_test_continuous >= 0.5, 1, 0)

    val_accuracy = accuracy_score(y_test, y_pred_test_class)
    val_f1 = f1_score(y_test, y_pred_test_class, average="weighted")
    val_recall = recall_score(y_test, y_pred_test_class, average="weighted")
    val_precision = precision_score(y_test, y_pred_test_class, average="weighted")
    
    train_accuracy = accuracy_score(y_train, y_pred_train_class)
    train_f1 = f1_score(y_train, y_pred_train_class, average="weighted")
    train_recall = recall_score(y_train, y_pred_train_class, average="weighted")
    train_precision = precision_score(y_train, y_pred_train_class, average="weighted")


    return {
        "val_accuracy": val_accuracy,
        "val_f1": val_f1,
        "val_recall": val_recall,
        "val_precision": val_precision,

        "train_accuracy": train_accuracy,
        "train_f1": train_f1,
        "train_recall": train_recall,
        "train_precision": train_precision,
    }

def main(data_file, scaler_file):
    warnings.filterwarnings("ignore")
    
    print("MLflow experiment set to 'Diabetes Classification - Skilled Tuning'")

    try:
        train_df, test_df = load_data(data_file)
        print(f"Data berhasil di-load dari {data_file}")
    except FileNotFoundError:
        print(f"File data tidak ditemukan.")
        return

    try:
        scaler = joblib.load(scaler_file)
        print(f"MinMax scaler berhasil di-load dari {scaler_file}.")

        scaled_train_array = scaler.transform(train_df)
        scaled_test_array = scaler.transform(test_df)

        scaled_train_df = pd.DataFrame(scaled_train_array, columns=train_df.columns)
        scaled_test_df = pd.DataFrame(scaled_test_array, columns=test_df.columns)

        target_col = 'Outcome'

        X_train_scaled = scaled_train_df.drop(target_col, axis=1)
        X_test_scaled = scaled_test_df.drop(target_col, axis=1)

        y_train = train_df[target_col].values 
        y_test = test_df[target_col].values
        
        print("Data train dan test telah di-scaling dan dipisahkan.")

    except FileNotFoundError:
        print(f"File '{scaler_file}' tidak ditemukan.")
        return
    except ValueError as e:
        print(f"Terjadi Value Error saat scaling: {e}")
        return

    param_grid = {
        'fit_intercept': [True, False],
        'positive': [True, False]
    }
    
    best_accuracy = 0
    best_params = {}
    best_metrics = None
    best_model = None

    print("hyperparameter tuning untuk linearRegression...")

    for fit_intercept in param_grid['fit_intercept']:
        for positive in param_grid['positive']:
            
            run_name = f"run_fit_intercept{fit_intercept}_positive{positive}"
            
            with mlflow.start_run(run_name=run_name, nested=True):
                params = {
                    "fit_intercept": fit_intercept, 
                    "positive": positive
                }
                mlflow.log_params(params)

                model = LinearRegression(
                    fit_intercept=fit_intercept, 
                    positive=positive
                )
                model.fit(X_train_scaled, y_train)

                metrics = hitung_metrik(
                    X_train_scaled, y_train,
                    X_test_scaled, y_test, model
                )

                mlflow.log_metrics(metrics)

                if metrics["val_accuracy"] > best_accuracy:
                    best_accuracy = metrics["val_accuracy"]
                    best_params = params
                    best_metrics = metrics
                    best_model = model

    print("\nHyperparameter Tuning Selesai.")
    
    if best_model is not None:
        print("Mencatat model terbaik ke MLflow sebagai run 'Best_Model'...")
        with mlflow.start_run(run_name="Best_Model"):
            
            mlflow.log_params(best_params)
            mlflow.log_metrics({"best_" + k: v for k, v in best_metrics.items()})
            
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="model"
            )

        print("\n=== HASIL TERBAIK ===")
        print("Best Params:", best_params)
        print("Best Validation Accuracy:", best_metrics.get('val_accuracy'))
        print("=======================")

    else:
        print("Tidak ada model yang berhasil dilatih.")
        
    print("\nRun complete. Jalankan 'mlflow ui' di terminal Anda untuk melihat hasilnya.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="diabetes_cleaned.csv")
    parser.add_argument("--scaler_file", type=str, default="minmax_scaler.joblib")
    args = parser.parse_args()
    
    main(args.data_file, args.scaler_file)
