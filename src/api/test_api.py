import time
from pathlib import Path

import numpy as np
import requests

# Ensure we can load data using relative paths correctly
project_root = Path(__file__).resolve().parents[2]
test_data_path = project_root / "data" / "processed" / "X_test.npy"


def test_api():
    base_url = "http://127.0.0.1:8000"

    print("Waiting for server to start...")
    max_retries = 10
    server_up = False
    for i in range(max_retries):
        try:
            r = requests.get(f"{base_url}/health")
            if r.status_code == 200:
                server_up = True
                break
        except requests.ConnectionError:
            time.sleep(2)

    if not server_up:
        print("FAIL: Could not connect to API server.")
        return

    # GET /health
    print("\nTesting GET /health")
    response = requests.get(f"{base_url}/health")
    if response.status_code == 200:
        print(f"PASS - Response: {response.json()}")
    else:
        print(f"FAIL - Status code: {response.status_code}")

    # GET /threshold
    print("\nTesting GET /threshold")
    response = requests.get(f"{base_url}/threshold")
    if response.status_code == 200:
        print(f"PASS - Response: {response.json()}")
    else:
        print(f"FAIL - Status code: {response.status_code}")

    # GET /metrics
    print("\nTesting GET /metrics")
    response = requests.get(f"{base_url}/metrics")
    if response.status_code == 200:
        print(f"PASS - Response: {response.json()}")
    else:
        print(f"FAIL - Status code: {response.status_code}")

    # POST /predict
    print("\nTesting POST /predict")
    try:
        X_test = np.load(test_data_path)
        # Sample 5 random windows
        indices = np.random.choice(len(X_test), 5, replace=False)
        sample_windows = X_test[indices].tolist()

        payload = {"windows": sample_windows}
        response = requests.post(f"{base_url}/predict", json=payload)

        if response.status_code == 200:
            print(f"PASS - Response: {response.json()}")
        else:
            print(f"FAIL - Status code: {response.status_code}\n{response.text}")
    except Exception as e:
        print(f"FAIL - Error loading test data or sending request: {e}")


if __name__ == "__main__":
    test_api()
