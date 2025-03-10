import torch
from torch import nn

import json
import pandas as pd
import numpy as np
from typing import Dict, Union, List

from sklearn.preprocessing import LabelEncoder

class ContextEncoder(nn.Module):
    def __init__(self, context_cols: Dict[str, str], config_path):
        super().__init__()
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.context_cols = context_cols
        self.embedding_dim = config["embedding_dim"]
        self.linear_layers = nn.ModuleDict()  # Use ModuleDict for learnable layers
        self.output_dim = 0
        self.encoded_features = []
        self.separation_index = {"numerical": 0, "time": 0}
        
        for col, col_type in context_cols.items():
            if col_type == 'numerical':
                self.linear_layers[col] = nn.Linear(1, self.embedding_dim)
                self.output_dim += self.embedding_dim
                self.encoded_features.append(col)
                self.separation_index[col_type] += 1

            elif col_type == 'time':
                # Still call _encode_time, but now use a learnable Linear layer
                self.linear_layers[col] = nn.Linear(1, self.embedding_dim)
                self.output_dim += self.embedding_dim
                self.encoded_features.append(col)

            elif col_type == 'categorical':
                # No categorical in the SIMPLIFIED example
                raise ValueError("Categorical encoding not Supported yet")
            else:
                raise ValueError(f"Invalid context type for {col}: {col_type}")

    def forward(self, x: torch.Tensor, device='cuda') -> torch.Tensor:
        encoded_context = []
        for col, col_type in self.context_cols.items():
            if col_type == 'numerical':
                values = torch.tensor(x[:self.separation_index['numerical'], :], dtype=torch.float32)
                encoded_context.append(self.linear_layers[col](values))

            elif col_type == 'time':
                encoded_time = _encode_time(x[self.separation_index['numerical']:, :].squeeze_(dim=0))
                encoded_time = encoded_time.to(device)
                encoded_context.append(self.linear_layers[col](encoded_time.unsqueeze_(dim=-1)))

        return torch.cat(encoded_context, dim=-1)

    def get_output_dim(self):
        return self.output_dim


def _encode_time(timestamp: Union[torch.Tensor, List[float]]) -> torch.Tensor:
    # Convert to datetime objects if they are not already (e.g., from timestamps)
    if isinstance(timestamp, list):
        timestamp = torch.tensor(timestamp, dtype=torch.float32)
    if not isinstance(timestamp, torch.Tensor):
        raise ValueError(f"Time context must be list or tensor")
     #Assumes input are number
    timestamp = pd.to_datetime(timestamp.cpu().numpy(), unit='s') # Convert to datetime objects
    timestamp = pd.DataFrame(timestamp, columns=["timestamp"])

    # Extract components
    hour = timestamp["timestamp"].dt.hour
    dayofweek = timestamp["timestamp"].dt.dayofweek
    dayofmonth = timestamp["timestamp"].dt.day
    month = timestamp["timestamp"].dt.month
    year = timestamp["timestamp"].dt.year

    # Cyclical features
    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)
    dayofmonth_sin = np.sin(2 * np.pi * dayofmonth / 31.0)  # Approximate days in month
    dayofmonth_cos = np.cos(2 * np.pi * dayofmonth / 31.0)

    # Combine features, making sure they're all float32 for PyTorch
    encoded_time = np.concatenate([
        hour_sin.values.reshape(-1, 1),
        hour_cos.values.reshape(-1, 1),
        dayofmonth_sin.values.reshape(-1, 1),
        dayofmonth_cos.values.reshape(-1, 1),
        year.values.reshape(-1, 1)
    ], axis=1).astype(np.float32)
    
    return torch.from_numpy(encoded_time) # Return as PyTorch tensor


def fit_categorical_encoders(context_data: Dict[str, List[str]],
                              context_cols: Dict[str, str]) -> Dict[str, 'LabelEncoder']:
    categorical_encoders = {}
    for col, col_type in context_cols.items():
        if col_type == 'categorical':
            if col not in context_data:
                raise ValueError(f"Categorical context column '{col}' not found in data.")
            encoder = LabelEncoder()
            encoder.fit(context_data[col])
            categorical_encoders[col] = encoder
    return categorical_encoders


if __name__ == '__main__':
    # Example Usage
    # context_data = {
    #     'location': torch.randn(10, 1),
    #     'time': torch.arange(1678886400, 1678886400 + 10, dtype=torch.float32),  # Example timestamps
    #     'solar_activity': torch.rand(10, 1),
    #     'season': ["spring", "summer", "fall", "winter", "spring", "summer", "fall", "winter", "spring", "summer"]
    # }
    # 
    # context_cols = {
    #     'location': 'numerical',
    #     'time': 'time',
    #     'solar_activity': 'numerical',
    #     'season': 'categorical'
    # }
    # 
    # encoder = ContextEncoder(
    #     context_cols,
    #     embedding_dim=32
    # )
    # 
    # embedding = encoder(context_data)
    #
    # print(embedding)
    
    pass