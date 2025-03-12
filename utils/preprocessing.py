import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from torch.utils.data import Dataset, DataLoader, Subset
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
                 header: str = "infer",
                 outlier_removal: bool = False,
                 outlier_method: str = 'clamp',
                 outlier_std_threshold: float = 2.0
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
        self.outlier_removal = outlier_removal
        self.outlier_method = outlier_method
        self.outlier_std_threshold = outlier_std_threshold


        # Load and preprocess data
        self.load_and_preprocess_data()
        self.shift_target()  # Shift the target variable
        self.add_previous_target()  # Add previous target as a feature


    def load_and_preprocess_data(self):
        """Loads and preprocesses the data."""
        self.load_data()
        self.validate_columns()
        self.preprocess_data()
        self.ensure_numerical_types()

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
            if col in self.data.columns:
                if method == 'standard':
                    scaler = StandardScaler()
                elif method == 'minmax':
                    scaler = MinMaxScaler()
                else:
                    raise ValueError(f"Invalid scaling method: {method}")
                self.data[col] = scaler.fit_transform(self.data[[col]])

        # 3. scale Target (before shifting and adding previous target)
        if self.norm_target:
            self.scaler_target = StandardScaler() # Store the scaler
            self.data[self.target_col] = self.scaler_target.fit_transform(self.data[[self.target_col]])


        # 4. One-Hot Encode Categorical Features
        if self.categorical_features:
            self.data = pd.get_dummies(self.data, columns=self.categorical_features, prefix=self.categorical_features, dummy_na=False)

            # Correctly update feature_cols
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

        for col in self.feature_cols + [self.target_col]:
            try:
                self.data[col] = pd.to_numeric(self.data[col], errors='raise').astype(np.float32)
            except ValueError as e:
                print(f"Error converting column '{col}' to numeric: {e}")
                print(f"Unique values in '{col}':", self.data[col].unique())
                raise

    def shift_target(self):
        """Shifts the target variable by one step ahead."""
        self.data[self.target_col] = self.data[self.target_col].shift(-1)
        self.data.dropna(subset=[self.target_col], inplace=True) # Drop the last row (NaN target)

    def add_previous_target(self):
        """Adds the previous target value as a feature."""
        self.data['prev_target'] = self.data[self.target_col].shift(1).fillna(method='bfill')
        self.feature_cols.append('prev_target')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
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



def clamp_outliers(data: pd.DataFrame, feature_cols: List[str], std_threshold: float) -> pd.DataFrame:
    """Clamps outliers to a specified standard deviation threshold."""
    data_clamped = data.copy()
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(data_clamped[col]):  # Only clamp numeric columns
            mean = data_clamped[col].mean()
            std = data_clamped[col].std()
            lower_bound = mean - std_threshold * std
            upper_bound = mean + std_threshold * std
            data_clamped[col] = np.clip(data_clamped[col], lower_bound, upper_bound)
    return data_clamped


def split_data(dataset: TabularDataset, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1, random_seed: int = 42) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Splits data into train, validation, and test sets.  Train/Val are shuffled,
    Test is sequential.  Returns Datasets.
    """
    dataset_size = len(dataset)
    test_size = int(test_ratio * dataset_size)
    train_val_size = dataset_size - test_size
    train_size = int(train_ratio / (train_ratio + val_ratio) * train_val_size)
    val_size = train_val_size - train_size

    # --- Create Indices ---
    train_val_indices = list(range(train_val_size))
    test_indices = list(range(train_val_size, dataset_size))

    # --- Shuffle Train/Val Indices ---
    rng = np.random.default_rng(random_seed)
    rng.shuffle(train_val_indices)

    train_indices = train_val_indices[:train_size]
    val_indices = train_val_indices[train_size:]

    # --- Create Subsets ---
    # Create Subsets *before* outlier removal
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # --- Outlier Removal (Train/Val ONLY) ---
    if dataset.outlier_removal:
        # Convert Subsets back to DataFrames (efficiently)
        train_df = dataset.data.iloc[train_indices].copy()
        val_df = dataset.data.iloc[val_indices].copy()

        # Apply clamping
        train_df_clamped = clamp_outliers(train_df, dataset.feature_cols + [dataset.target_col], dataset.outlier_std_threshold)
        val_df_clamped = clamp_outliers(val_df, dataset.feature_cols + [dataset.target_col], dataset.outlier_std_threshold)

        # Create new *Datasets* from the clamped DataFrames.  This is now MUCH cleaner.
        train_dataset = CustomDataFrameDataset(train_df_clamped, dataset)
        val_dataset = CustomDataFrameDataset(val_df_clamped, dataset)
        # test_dataset remains unchanged, as we don't remove outliers from it.

    return train_dataset, val_dataset, test_dataset  # Return Subsets


class CustomDataFrameDataset(Dataset):
    """
    A custom Dataset that wraps a DataFrame but inherits metadata from the original TabularDataset.
    This keeps all the preprocessing and feature information consistent.
    """
    def __init__(self, dataframe: pd.DataFrame, original_dataset: TabularDataset):
        self.data = dataframe
        self.feature_cols = original_dataset.feature_cols
        self.target_col = original_dataset.target_col
        self.context_cols = original_dataset.context_cols
        # You might need to store other attributes from original_dataset if needed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        features = torch.tensor(row[self.feature_cols].values, dtype=torch.float32)
        target = torch.tensor(row[self.target_col], dtype=torch.float32)

        context = {}
        if self.context_cols:
            for col in self.context_cols:
                if pd.api.types.is_numeric_dtype(row[col]): # Check row data type
                   context[col] = torch.tensor(row[col], dtype=torch.float32)
                else:
                   context[col] = str(row[col])

        return {'features': features, 'target': target, 'context': context}


def create_dataloader(dataset: Dataset, batch_size: int = 32, shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)