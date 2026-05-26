import numpy as np
import pytest

from src.data.preprocess import create_windows


def test_create_windows_shape():
    # Create dummy data: 100 timesteps, 8 features
    data = np.random.rand(100, 8)

    # Create windows with default size 30, stride 1
    windows = create_windows(data, window_size=30, stride=1)

    # Expected shape: (100 - 30 + 1, 30, 8) = (71, 30, 8)
    assert windows.shape == (71, 30, 8)


def test_create_windows_with_labels():
    # Create dummy data
    data = np.random.rand(50, 8)
    labels = np.zeros(50)

    # Introduce an anomaly at index 20
    labels[20] = 1

    windows, window_labels = create_windows(data, labels=labels, window_size=10, stride=1)

    # Expected shape: (50 - 10 + 1, 10, 8) = (41, 10, 8)
    assert windows.shape == (41, 10, 8)
    assert window_labels.shape == (41,)

    # The anomaly is at index 20.
    # Window 0 covers indices 0 to 9.
    # Window 11 covers indices 11 to 20 (contains anomaly).
    # Window 20 covers indices 20 to 29 (contains anomaly).
    # Window 21 covers indices 21 to 30 (no anomaly).

    assert window_labels[10] == 0
    assert window_labels[11] == 1
    assert window_labels[20] == 1
    assert window_labels[21] == 0


def test_skab_schema_valid():
    import pandas as pd

    from src.data.preprocess import SKAB_SCHEMA

    # Valid data matching the schema
    valid_data = pd.DataFrame(
        {
            "Accelerometer1RMS": [0.1, 0.2],
            "Accelerometer2RMS": [0.3, 0.4],
            "Current": [1.5, 1.6],
            "Pressure": [2.1, 2.2],
            "Temperature": [35.0, 36.0],
            "Thermocouple": [25.0, 26.0],
            "Voltage": [220.0, 221.0],
            "Volume Flow RateRMS": [10.0, 11.0],
            "anomaly": [0, 1],
        }
    )

    validated = SKAB_SCHEMA.validate(valid_data)
    assert validated.shape == (2, 9)


def test_skab_schema_missing_column():
    import pandas as pd
    import pandera.pandas as pa

    from src.data.preprocess import SKAB_SCHEMA

    # Missing 'Current' column
    invalid_data = pd.DataFrame(
        {
            "Accelerometer1RMS": [0.1],
            "Accelerometer2RMS": [0.3],
            "Pressure": [2.1],
            "Temperature": [35.0],
            "Thermocouple": [25.0],
            "Voltage": [220.0],
            "Volume Flow RateRMS": [10.0],
            "anomaly": [0],
        }
    )

    with pytest.raises(pa.errors.SchemaError):
        SKAB_SCHEMA.validate(invalid_data)


def test_skab_schema_invalid_label():
    import pandas as pd
    import pandera.pandas as pa

    from src.data.preprocess import SKAB_SCHEMA

    # Invalid value '3' in 'anomaly' column (must be 0 or 1)
    invalid_data = pd.DataFrame(
        {
            "Accelerometer1RMS": [0.1],
            "Accelerometer2RMS": [0.3],
            "Current": [1.5],
            "Pressure": [2.1],
            "Temperature": [35.0],
            "Thermocouple": [25.0],
            "Voltage": [220.0],
            "Volume Flow RateRMS": [10.0],
            "anomaly": [3],
        }
    )

    with pytest.raises(pa.errors.SchemaError):
        SKAB_SCHEMA.validate(invalid_data)
