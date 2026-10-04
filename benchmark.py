import time
import json
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import lightgbm as lgb

DATA_PATH = os.path.expanduser("~/ml-benchmark/creditcard.csv")
RESULT_PATH = "benchmark_result.json"

def main():
    print("=== LightGBM Benchmark: Credit Card Fraud Detection ===")
    
    # 1. Load dataset and measure time
    start_load = time.time()
    if not os.path.exists(DATA_PATH):
        # Fallback to local directory if exists
        local_fallback = "creditcard.csv"
        if os.path.exists(local_fallback):
            data_file = local_fallback
        else:
            raise FileNotFoundError(f"Dataset not found at {DATA_PATH} or {local_fallback}. Please download creditcard.csv first.")
    else:
        data_file = DATA_PATH
        
    print(f"Loading dataset from {data_file}...")
    df = pd.read_csv(data_file)
    load_time = time.time() - start_load
    print(f"Data loaded in {load_time:.4f} seconds. Shape: {df.shape}")
    
    # Features and Target
    X = df.drop(columns=['Class'])
    y = df['Class']
    
    # Train / Test Split (80% train, 20% test, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 2. Train LGBMClassifier and measure training time
    print("Training LightGBM model...")
    start_train = time.time()
    model = lgb.LGBMClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
    )
    train_time = time.time() - start_train
    best_iteration = getattr(model, "best_iteration_", model.n_estimators)
    print(f"Training completed in {train_time:.4f} seconds. Best iteration: {best_iteration}")
    
    # 3. Evaluate on Test Set
    print("Evaluating model on test set...")
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    auc_roc = float(roc_auc_score(y_test, y_pred_prob))
    accuracy = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    
    print(f"Metrics -> AUC-ROC: {auc_roc:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    
    # 4. Measure Inference Latency (1 row) and Throughput (batch of 1000 rows)
    print("Measuring inference performance...")
    single_sample = X_test.iloc[[0]]
    
    # Warmup
    model.predict(single_sample)
    
    # 1 row latency (average over 100 runs)
    latencies = []
    for _ in range(100):
        t0 = time.time()
        model.predict(single_sample)
        latencies.append((time.time() - t0) * 1000.0) # in ms
    avg_latency_ms = float(np.mean(latencies))
    
    # Throughput (1000 rows batch)
    batch_sample = X_test.iloc[:1000]
    t0 = time.time()
    model.predict(batch_sample)
    batch_time = time.time() - t0
    throughput = float(len(batch_sample) / batch_time) if batch_time > 0 else 0.0
    
    print(f"Inference Latency (1 row): {avg_latency_ms:.4f} ms")
    print(f"Inference Throughput (1000 rows batch): {throughput:.2f} rows/sec")
    
    # 5. Compile Results
    results = {
        "load_time_sec": round(load_time, 4),
        "train_time_sec": round(train_time, 4),
        "best_iteration": int(best_iteration),
        "auc_roc": round(auc_roc, 4),
        "accuracy": round(accuracy, 4),
        "f1_score": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "inference_latency_1_row_ms": round(avg_latency_ms, 4),
        "inference_throughput_1000_rows_sec": round(throughput, 2)
    }
    
    # 6. Save to benchmark_result.json
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    print(f"Results successfully saved to {RESULT_PATH}")

if __name__ == "__main__":
    main()
