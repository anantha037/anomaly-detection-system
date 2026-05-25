import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score
from pathlib import Path

class ThresholdEngine:
    def __init__(self, model, device, scaler_path):
        self.model = model
        self.device = device
        self.scaler = joblib.load(scaler_path)
        self.model.to(self.device)
        
    def compute_errors(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        errors = []
        batch_size = 256
        
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch_x = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(self.device)
                preds = self.model(batch_x)
                
                # Compute MSE per window
                batch_errors = torch.mean((preds - batch_x)**2, dim=(1, 2)).cpu().numpy()
                errors.append(batch_errors)
                
        return np.concatenate(errors)
        
    def tune_threshold(self, train_errors, multipliers=[2.0, 2.5, 3.0, 3.5, 4.0]) -> dict:
        mean_err = np.mean(train_errors)
        std_err = np.std(train_errors)
        
        thresholds = {}
        for k in multipliers:
            thresholds[k] = mean_err + k * std_err
            
        return thresholds
        
    def find_best_threshold(self, X_test, y_test, train_errors, multipliers):
        test_errors = self.compute_errors(X_test)
        thresholds = self.tune_threshold(train_errors, multipliers)
        
        metrics = []
        best_f1 = -1
        best_multiplier = None
        best_threshold = None
        
        for k, threshold in thresholds.items():
            y_pred = (test_errors > threshold).astype(int)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            metrics.append({
                "multiplier": k,
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1
            })
            
            if f1 > best_f1:
                best_f1 = f1
                best_multiplier = k
                best_threshold = threshold
                
        metrics_df = pd.DataFrame(metrics)
        return best_multiplier, best_threshold, metrics_df
        
    def save_threshold(self, threshold, path):
        np.save(path, threshold)
