"""
Data Preprocessor for LSTM Price Direction Prediction

Handles data loading, normalization, and sequence generation using sliding windows.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Tuple, Optional, List
import torch
from torch.utils.data import Dataset, DataLoader


class PriceDirectionDataset(Dataset):
    """PyTorch Dataset for price direction prediction."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: Input sequences of shape (n_samples, sequence_length, n_features)
            y: Target labels of shape (n_samples,)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class DataPreprocessor:
    """
    Preprocesses OHLCV data for LSTM price direction prediction.
    
    Features:
    - Min-Max or Z-score normalization
    - Sliding window sequence generation
    - Time series train/test split
    - Binary label generation for price direction
    """
    
    def __init__(
        self,
        sequence_length: int = 60,
        prediction_horizon: int = 1,
        normalization: str = 'minmax',
        feature_columns: Optional[List[str]] = None
    ):
        """
        Args:
            sequence_length: Number of time steps to look back
            prediction_horizon: Number of steps ahead to predict
            normalization: 'minmax' or 'zscore'
            feature_columns: List of column names to use as features
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.normalization = normalization
        self.feature_columns = feature_columns or ['open', 'high', 'low', 'close', 'volume']
        
        if normalization == 'minmax':
            self.scaler = MinMaxScaler()
        elif normalization == 'zscore':
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown normalization method: {normalization}")
        
        self.is_fitted = False
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load OHLCV data from CSV file.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame with OHLCV data
        """
        df = pd.read_csv(filepath)
        
        # Ensure timestamp column exists and is properly formatted
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def create_labels(self, df: pd.DataFrame, price_column: str = 'close') -> np.ndarray:
        """
        Create binary labels for price direction.
        
        Args:
            df: DataFrame with price data
            price_column: Column name for price (default: 'close')
            
        Returns:
            Binary labels (1 for up, 0 for down)
        """
        prices = df[price_column].values
        
        # Calculate future price change
        future_prices = np.roll(prices, -self.prediction_horizon)
        
        # Create binary labels: 1 if price goes up, 0 if down
        labels = (future_prices > prices).astype(int)
        
        # Remove last prediction_horizon samples (no future data)
        labels = labels[:-self.prediction_horizon]
        
        return labels
    
    def normalize_data(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Normalize feature data.
        
        Args:
            df: DataFrame with features
            fit: Whether to fit the scaler (True for training data)
            
        Returns:
            Normalized feature array
        """
        features = df[self.feature_columns].values
        
        if fit:
            normalized = self.scaler.fit_transform(features)
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise ValueError("Scaler must be fitted before transform")
            normalized = self.scaler.transform(features)
        
        return normalized
    
    def create_sequences(
        self,
        data: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding window sequences for LSTM input.
        
        Args:
            data: Normalized feature array
            labels: Binary labels
            
        Returns:
            Tuple of (sequences, labels)
            - sequences: shape (n_samples, sequence_length, n_features)
            - labels: shape (n_samples,)
        """
        X, y = [], []
        
        # Create sequences using sliding window
        for i in range(len(data) - self.sequence_length - self.prediction_horizon + 1):
            X.append(data[i:i + self.sequence_length])
            y.append(labels[i + self.sequence_length - 1])
        
        return np.array(X), np.array(y)
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        fit_scaler: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete data preparation pipeline.
        
        Args:
            df: DataFrame with OHLCV data
            fit_scaler: Whether to fit the scaler
            
        Returns:
            Tuple of (X, y) ready for model training
        """
        # Create labels
        labels = self.create_labels(df)
        
        # Normalize features
        normalized_data = self.normalize_data(df, fit=fit_scaler)
        
        # Remove last prediction_horizon samples to align with labels
        normalized_data = normalized_data[:-self.prediction_horizon]
        
        # Create sequences
        X, y = self.create_sequences(normalized_data, labels)
        
        return X, y
    
    def train_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train and test sets while preserving time order.
        
        Args:
            X: Input sequences
            y: Labels
            test_size: Proportion of data for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def create_dataloaders(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        batch_size: int = 32,
        shuffle_train: bool = False
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders for training and validation.
        
        Args:
            X_train: Training sequences
            y_train: Training labels
            X_val: Validation sequences
            y_val: Validation labels
            batch_size: Batch size
            shuffle_train: Whether to shuffle training data (not recommended for time series)
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        train_dataset = PriceDirectionDataset(X_train, y_train)
        val_dataset = PriceDirectionDataset(X_val, y_val)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        return train_loader, val_loader
    
    def get_feature_dim(self) -> int:
        """Get the number of features."""
        return len(self.feature_columns)
