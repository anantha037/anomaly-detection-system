import torch
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, data):
        """
        Args:
            data (np.ndarray): Array of shape [num_windows, window_size, num_features]
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        # Autoencoder: input and target are the same
        window = self.data[idx]
        return window, window

def get_dataloader(data, batch_size=32, shuffle=True):
    """
    Helper function to create a DataLoader from numpy data.
    """
    dataset = TimeSeriesDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
