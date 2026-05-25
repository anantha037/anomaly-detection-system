import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class Alert:
    window_id: int
    timestamp: str
    reconstruction_error: float
    threshold: float
    is_anomaly: bool
    severity: str

class AlertSystem:
    def __init__(self, threshold):
        self.threshold = threshold
        self.total_windows = 0
        
    def generate_alerts(self, errors: np.ndarray, timestamps=None) -> List[Alert]:
        self.total_windows = len(errors)
        alerts = []
        for i, error in enumerate(errors):
            if error > self.threshold:
                if timestamps is None:
                    timestamp = f"window_{i}"
                else:
                    timestamp = timestamps[i]
                
                # Severity logic
                if error < 1.2 * self.threshold:
                    severity = "low"
                elif error < 1.5 * self.threshold:
                    severity = "medium"
                else:
                    severity = "high"
                    
                alert = Alert(
                    window_id=i,
                    timestamp=timestamp,
                    reconstruction_error=float(error),
                    threshold=float(self.threshold),
                    is_anomaly=True,
                    severity=severity
                )
                alerts.append(alert)
                
        return alerts
        
    def summary(self, alerts: List[Alert]) -> dict:
        high = sum(1 for a in alerts if a.severity == "high")
        medium = sum(1 for a in alerts if a.severity == "medium")
        low = sum(1 for a in alerts if a.severity == "low")
        total = len(alerts)
        
        return {
            "total_alerts": total,
            "high_severity": high,
            "medium_severity": medium,
            "low_severity": low,
            "alert_rate": total / self.total_windows if self.total_windows > 0 else 0.0
        }
