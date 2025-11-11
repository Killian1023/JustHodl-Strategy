"""
LSTM to Predict Profit Opportunities

New Approach:
- OLD: Predict if next 15-min goes up or down (direction)
- NEW: Predict if there's a >0.2% profit opportunity in next 5 periods

Why this is better:
1. Directly targets trading goal (cover 0.2% commission)
2. More actionable signals
3. Better aligned with actual profit/loss

Label definition:
y = 1 if max(price[t+1:t+6]) > price[t] * 1.002 else 0
(if any future price within 5 periods is >0.2% higher)
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, WeightedRandomSampler
from .data_preprocessor import PriceDirectionDataset
from .evaluator import ModelEvaluator
from .backtester import TradingBacktester
from .technical_indicators import add_technical_indicators, get_selected_features
from sklearn.preprocessing import RobustScaler
import os


class ProfitOpportunityPreprocessor:
    """Preprocessor for profit opportunity prediction."""
    
    def __init__(
        self, 
        sequence_length: int = 60,
        forecast_periods: int = 5,
        profit_threshold: float = 0.002,  # 0.2%
        verbose: bool = True
    ):
        self.sequence_length = sequence_length
        self.forecast_periods = forecast_periods
        self.profit_threshold = profit_threshold
        self.scaler = RobustScaler()
        self.feature_columns = get_selected_features()
        self.is_fitted = False
        self.verbose = verbose
    
    def prepare_data(self, df: pd.DataFrame, fit_scaler: bool = True):
        """
        Prepare data with new labeling strategy.
        
        Label: 1 if any of next N periods has return > threshold, else 0
        """
        # Add technical indicators
        if self.verbose:
            print("Calculating technical indicators...")
        df = add_technical_indicators(df)
        
        # Get features FIRST (before removing samples)
        if self.verbose:
            print(f"\nCalculating features...")
        features = df[self.feature_columns].values
        prices = df['close'].values
        
        # Normalize features
        if fit_scaler:
            features = self.scaler.fit_transform(features)
            self.is_fitted = True
        else:
            features = self.scaler.transform(features)
        
        # Create sequences and labels
        # Key: For sequence ending at index i, predict future of prices[i]
        X, y = [], []
        
        if self.verbose:
            print(f"\nCreating sequences with aligned labels")
            print(f"  Sequence length: {self.sequence_length}")
            print(f"  Forecast periods: {self.forecast_periods} (next {self.forecast_periods * 15} minutes)")
            print(f"  Profit threshold: {self.profit_threshold*100:.1f}%")
        
        for i in range(self.sequence_length, len(prices) - self.forecast_periods):
            # Sequence: features from [i-sequence_length : i]
            sequence = features[i - self.sequence_length : i]
            
            # Label: based on future of prices[i]
            current_price = prices[i]
            future_prices = prices[i+1 : i+1+self.forecast_periods]
            max_future_price = future_prices.max()
            max_return = (max_future_price - current_price) / current_price
            
            label = 1 if max_return > self.profit_threshold else 0
            
            X.append(sequence)
            y.append(label)
        
        return np.array(X), np.array(y)
    
    def get_feature_dim(self):
        return len(self.feature_columns)


class ProfitOpportunityLSTM(nn.Module):
    """LSTM for profit opportunity prediction."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 96, dropout: float = 0.2):
        super(ProfitOpportunityLSTM, self).__init__()
        
        # 2-layer LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # FC layers
        self.fc1 = nn.Linear(hidden_dim, 48)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(48, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for name, param in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
            elif 'fc' in name:
                if 'weight' in name and param.dim() >= 2:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
            elif 'layer_norm' in name:
                if 'weight' in name:
                    nn.init.constant_(param, 1.0)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
    
    def forward(self, x):
        """Forward pass - returns logits."""
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        
        out = self.layer_norm(last_output)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

def train_epoch(model, train_loader, optimizer, criterion, device):
    """Training epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X, y in train_loader:
        X, y = X.to(device), y.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        correct += (preds == y).sum().item()
        total += y.size(0)
    
    return total_loss / len(train_loader), correct / total


def validate(model, val_loader, criterion, device):
    """Validation."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_probs = []
    
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device).unsqueeze(1)
            
            logits = model(X)
            loss = criterion(logits, y)
            
            total_loss += loss.item()
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy())
            
            preds = (probs >= 0.5).float()
            correct += (preds == y).sum().item()
            total += y.size(0)
    
    all_probs = np.array(all_probs)
    prob_stats = {
        'min': all_probs.min(),
        'max': all_probs.max(),
        'mean': all_probs.mean(),
        'std': all_probs.std()
    }
    
    return total_loss / len(val_loader), correct / total, prob_stats


def main():
    print("="*70)
    print("PROFIT OPPORTUNITY PREDICTION MODEL")
    print("="*70)
    print("\nGoal: Predict if there's a >0.2% profit opportunity")
    print("      in the next 5 periods (75 minutes)")
    
    # Configuration
    DATA_PATH = 'data/binance_BTCUSDT_15m_2023-01-01_to_2025-09-30.csv'
    SEQUENCE_LENGTH = 60
    FORECAST_PERIODS = 5  # Look ahead 5 periods (75 minutes)
    PROFIT_THRESHOLD = 0.002  # 0.2% profit opportunity
    BATCH_SIZE = 64
    EPOCHS = 80
    LEARNING_RATE = 0.001
    HIDDEN_DIM = 96
    DROPOUT = 0.2
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}\n")
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"Loaded {len(df)} data points")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}\n")
    
    # Prepare data with NEW labeling strategy
    preprocessor = ProfitOpportunityPreprocessor(
        sequence_length=SEQUENCE_LENGTH,
        forecast_periods=FORECAST_PERIODS,
        profit_threshold=PROFIT_THRESHOLD
    )
    
    X, y = preprocessor.prepare_data(df, fit_scaler=True)
    
    print(f"\nCreated {len(X)} sequences")
    print(f"Input shape: {X.shape}")
    print(f"Features: {preprocessor.get_feature_dim()}")
    
    # Class balance
    profit_opps = y.sum()
    no_profit = len(y) - profit_opps
    print(f"\nLabel distribution:")
    print(f"  Profit Opportunity (1): {profit_opps} ({profit_opps/len(y)*100:.1f}%)")
    print(f"  No Opportunity (0):     {no_profit} ({no_profit/len(y)*100:.1f}%)")
    
    if profit_opps / len(y) < 0.2:
        print("\n⚠️  Warning: Very few positive samples (<20%)")
        print("   Consider lowering profit threshold to 0.15% or 0.18%")
    elif profit_opps / len(y) > 0.6:
        print("\n⚠️  Warning: Too many positive samples (>60%)")
        print("   Consider raising profit threshold to 0.25% or 0.3%")
    else:
        print("\n✅ Label balance looks good (20-60% positive)")
    
    # Split data
    train_size = int(len(X) * 0.6)
    val_size = int(len(X) * 0.2)
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    print(f"\nData split:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples\n")
    
    # Create data loaders with balanced sampling
    class_counts = np.bincount(y_train.astype(int))
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[y_train.astype(int)]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    train_dataset = PriceDirectionDataset(X_train, y_train)
    val_dataset = PriceDirectionDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Create model
    print("Creating model...")
    model = ProfitOpportunityLSTM(
        input_dim=preprocessor.get_feature_dim(),
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT
    ).to(device)
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Loss and optimizer with class weighting
    pos_weight = torch.tensor([class_counts[0] / class_counts[1]]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.001
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    print(f"Using BCEWithLogitsLoss with pos_weight={pos_weight.item():.3f}")
    print("Starting training...")
    print("-"*70)
    
    # Training loop
    best_val_loss = float('inf')
    best_val_acc = 0
    patience = 15
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, prob_stats = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), './models/profit_opportunity_best.pth')
        else:
            patience_counter += 1
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"Prob: [{prob_stats['min']:.3f}, {prob_stats['max']:.3f}]")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    print("-"*70)
    print(f"Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    # Load best model and evaluate on test set
    print("\n" + "="*70)
    print("EVALUATING ON TEST SET")
    print("="*70)
    
    model.load_state_dict(torch.load('./models/profit_opportunity_best.pth'))
    model.eval()
    
    # Get predictions
    all_probs = []
    all_preds = []
    all_labels = []
    
    test_dataset = PriceDirectionDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            logits = model(X)
            probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            preds = (probs >= 0.5).astype(int)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    
    print(f"\nTest Set Performance:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f} (when predicting opportunity, how often correct)")
    print(f"  Recall:    {recall:.4f} (of all opportunities, how many caught)")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  TN: {cm[0,0]:5d}  FP: {cm[0,1]:5d}")
    print(f"  FN: {cm[1,0]:5d}  TP: {cm[1,1]:5d}")
    
    print(f"\nProbability distribution:")
    print(f"  Min:  {all_probs.min():.4f}")
    print(f"  Max:  {all_probs.max():.4f}")
    print(f"  Mean: {all_probs.mean():.4f}")
    print(f"  Std:  {all_probs.std():.4f}")
    
    # Analysis by confidence level
    print(f"\nPredictions by confidence level:")
    for threshold in [0.6, 0.7, 0.8, 0.9]:
        high_conf = all_probs >= threshold
        if high_conf.sum() > 0:
            high_conf_acc = (all_preds[high_conf] == all_labels[high_conf]).mean()
            print(f"  Prob >= {threshold}: {high_conf.sum():5d} samples, Accuracy: {high_conf_acc:.4f}")
    
    # Save results
    results = {
        'model': 'profit_opportunity',
        'profit_threshold': PROFIT_THRESHOLD,
        'forecast_periods': FORECAST_PERIODS,
        'test_accuracy': float(accuracy),
        'test_precision': float(precision),
        'test_recall': float(recall),
        'test_f1': float(f1),
        'test_auc': float(auc),
        'positive_rate': float(profit_opps / len(y)),
        'confusion_matrix': cm.tolist()
    }
    
    import json
    os.makedirs('./models', exist_ok=True)
    with open('./models/profit_opportunity_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Model saved to: ./models/profit_opportunity_best.pth")
    print(f"✅ Results saved to: ./models/profit_opportunity_results.json")
    
    # Trading implications
    print("\n" + "="*70)
    print("TRADING IMPLICATIONS")
    print("="*70)
    
    print(f"\nIf model accuracy is {accuracy:.1%}:")
    print(f"  - When it predicts opportunity → {precision:.1%} chance of being right")
    print(f"  - It catches {recall:.1%} of all actual opportunities")
    
    if precision > 0.55:
        print(f"\n✅ Precision >{0.55:.0%} is promising for trading!")
        print(f"   With stop-loss at 0.15% and take-profit at 0.3%:")
        print(f"   Expected value per trade ≈ {(precision * 0.3 - (1-precision) * 0.15):.2f}%")
    else:
        print(f"\n⚠️  Precision <55% may not be enough for profitable trading")
        print(f"   Consider:")
        print(f"   1. Training longer")
        print(f"   2. Adjusting profit threshold")
        print(f"   3. Using higher confidence threshold (e.g., prob > 0.7)")


if __name__ == '__main__':
    main()
