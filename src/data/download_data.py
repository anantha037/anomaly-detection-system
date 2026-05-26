import os
from pathlib import Path

import requests


def download_data():
    base_url = "https://raw.githubusercontent.com/waico/SKAB/master/data"
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)

    files_to_download = {
        "anomaly-free.csv": f"{base_url}/anomaly-free/anomaly-free.csv",
    }

    for i in range(1, 5):
        files_to_download[f"valve1_{i}.csv"] = f"{base_url}/valve1/{i}.csv"
    for i in range(1, 4):
        files_to_download[f"valve2_{i}.csv"] = f"{base_url}/valve2/{i}.csv"

    downloaded_count = 0
    for filename, url in files_to_download.items():
        filepath = data_dir / filename
        print(f"Downloading {filename} from {url}...")
        response = requests.get(url)
        response.raise_for_status()

        with open(filepath, "wb") as f:
            f.write(response.content)

        size = os.path.getsize(filepath)
        print(f"Successfully downloaded {filename}. Size: {size / 1024:.2f} KB")
        downloaded_count += 1

    print(f"Total files downloaded: {downloaded_count}")


if __name__ == "__main__":
    download_data()
