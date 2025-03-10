import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple
import numpy as np  # Import numpy explicitly

class TabularDataset(Dataset):

    def __init__(self,
                 data_path: str,
                 feature_cols: List[str],
                 target_col: str,
                 norm_target: bool = True,
                 context_cols: Optional[List[str]] = None,
                 file_type: str = 'csv',
                 numerical_features_info: Dict[str, str] = None,
                 categorical_features: List[str] = None,
                 sep: str = ",",
                 header: str = "infer"
                 ):
        super().__init__()

        self.data_path = data_path
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.norm_target = norm_target
        self.context_cols = context_cols if context_cols is not None else []
        self.file_type = file_type.lower()
        self.numerical_features_info = numerical_features_info if numerical_features_info is not None else {}
        self.categorical_features = categorical_features if categorical_features is not None else []
        self.sep = sep
        self.header = header

        # Load and preprocess data
        self.load_and_preprocess_data()


    def load_and_preprocess_data(self):
        """Loads and preprocesses the data in a single, unified method."""
        self.load_data()
        self.validate_columns()
        self.preprocess_data()
        self.ensure_numerical_types() # Added: Ensure all features are numeric


    def load_data(self):
        """Loads data from the specified file."""
        if self.file_type == 'csv':
            self.data = pd.read_csv(self.data_path, sep=self.sep, header=self.header)
        elif self.file_type == 'parquet':
            self.data = pd.read_parquet(self.data_path)
        else:
            raise ValueError(f"Unsupported file type: {self.file_type}")


    def validate_columns(self):
        """Validates that specified columns exist in the data."""
        all_cols = self.feature_cols + [self.target_col] + self.context_cols
        if not all(col in self.data.columns for col in all_cols):
            missing_cols = [col for col in all_cols if col not in self.data.columns]
            raise ValueError(f"Columns not found in data: {missing_cols}")

    def preprocess_data(self):
        """Preprocesses the data (handling missing values, scaling, encoding)."""

        # 1. Handle Missing Values (Imputation)
        for col in self.data.columns:
            if self.data[col].isnull().any():
                if col in self.numerical_features_info:
                    # Numerical: Impute with median
                    self.data[col] = self.data[col].fillna(self.data[col].median())
                elif col in self.categorical_features + [self.target_col] + self.context_cols:
                    # Categorical: Impute with mode
                    self.data[col] = self.data[col].fillna(self.data[col].mode()[0])

        # 2. Scale Numerical Features
        for col, method in self.numerical_features_info.items():
            if col in self.data.columns:  # Check if the column still exists
                if method == 'standard':
                    scaler = StandardScaler()
                elif method == 'minmax':
                    scaler = MinMaxScaler()
                else:
                    raise ValueError(f"Invalid scaling method: {method}")
                # Reshape, fit_transform, and ensure float32
                self.data[col] = scaler.fit_transform(self.data[[col]])
                
        # 3. scale Target
        if self.norm_target:
            self.data[self.target_col] = StandardScaler().fit_transform(self.data[[self.target_col]])

        # 4. One-Hot Encode Categorical Features
        if self.categorical_features:
            self.data = pd.get_dummies(self.data, columns=self.categorical_features, prefix=self.categorical_features, dummy_na=False)

            # Correctly update feature_cols (as before)
            new_feature_cols = []
            for col in self.feature_cols:
                if col not in self.categorical_features:
                    new_feature_cols.append(col)
            for col in self.data.columns:
                if any(cat_col in col for cat_col in self.categorical_features):
                    new_feature_cols.append(col)
            self.feature_cols = new_feature_cols

    def ensure_numerical_types(self):
        """Ensures all feature and target columns are numeric (float32)."""

        # Convert feature columns to float32
        for col in self.feature_cols:
            # Explicitly attempt numeric conversion, handle errors
            try:
                self.data[col] = pd.to_numeric(self.data[col], errors='raise').astype(np.float32)
            except ValueError as e:
                print(f"Error converting column '{col}' to numeric: {e}")
                print(f"Unique values in '{col}':", self.data[col].unique())
                raise

        # Convert target column to float32
        try:
            self.data[self.target_col] = pd.to_numeric(self.data[self.target_col], errors='raise').astype(np.float32)
        except ValueError as e:
            print(f"Error converting target column '{self.target_col}' to numeric: {e}")
            print(f"Unique values in '{self.target_col}':", self.data[self.target_col].unique())
            raise


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # No changes needed here, BUT the CRITICAL part is that
        # self.data is guaranteed to have the correct types by now.
        features = torch.tensor(row[self.feature_cols].values, dtype=torch.float32)
        target = torch.tensor(row[self.target_col], dtype=torch.float32)

        context = {}
        if self.context_cols:
            for col in self.context_cols:
                if pd.api.types.is_numeric_dtype(self.data[col]):
                    context[col] = torch.tensor(row[col], dtype=torch.float32)
                else:
                    context[col] = str(row[col])  # Keep as string

        return {'features': features, 'target': target, 'context': context}

def split_data(dataset: TabularDataset, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1,
               random_seed: int = 42) -> Tuple[Dataset, Dataset, Dataset]:
    if train_ratio + val_ratio + test_ratio != 1.0:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")

    dataset_size = len(dataset)

    # --- Method 1: Adjust Split Sizes (Robust for Small Datasets) ---
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size

    # Ensure at least one sample in each split
    if train_size == 0:
        train_size = 1
        if test_size > 0:
            test_size -=1
        elif val_size > 0:
            val_size -= 1
    if val_size == 0:
        val_size = 1
        if test_size > 0:
            test_size -= 1
        elif train_size > 0:
            train_size -= 1
    if test_size == 0:
        test_size = 1
        if val_size > 0:
            val_size -= 1
        elif train_size >0:
            train_size -= 1

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(random_seed)
    )
    return train_dataset, val_dataset, test_dataset

def create_dataloader(dataset: Dataset, batch_size: int = 32, shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


if __name__ == '__main__':
    # Create a larger dummy CSV file for demonstration
    data = {
        'sensor1': [1.0, 2.0, 3.0, 4.0, 5.0, "6", 7.0, 8.0, 9.0, 10.0] * 5,  # Introduce string
        'sensor2': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] * 5,
        'category': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'A', 'B', 'C'] * 5,
        'time': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 5,
        'target': [10.0, 12.0, 11.0, 13.0, 12.5, 11.5, 14.0, 13.5, 12.8, 11.2] * 5
    }
    df = pd.DataFrame(data)
    df.to_csv('dummy_data.csv', index=False)
    
    feature_cols = ['sensor1', 'sensor2', 'category']
    target_col = 'target'
    context_cols = ['time']
    numerical_features_info = {'sensor1': 'standard', 'sensor2': 'minmax'}
    categorical_features = ['category']

    dataset = TabularDataset(
        data_path='dummy_data.csv',
        feature_cols=feature_cols,
        target_col=target_col,
        context_cols=context_cols,
        numerical_features_info=numerical_features_info,
        categorical_features=categorical_features
    )


    train_dataset, val_dataset, test_dataset = split_data(dataset)
    train_loader = create_dataloader(train_dataset, batch_size=8)
    val_loader = create_dataloader(val_dataset)
    test_loader = create_dataloader(test_dataset)



    for batch in train_loader:
        features = batch['features']
        targets = batch['target']
        context = batch['context']
        print("Features:", features.dtype)
        print("Context:", context)
        print("Targets:", targets)
        break
    print(dataset.data.dtypes)