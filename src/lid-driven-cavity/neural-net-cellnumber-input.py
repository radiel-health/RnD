import numpy as np
import pandas as pd
import glob
import os
import random
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Define paths
DATA_DIR = Path(__file__).parent.parent.parent / "Data" / "results"

# Find all Re folders
re_folders = sorted([d for d in DATA_DIR.iterdir() if d.is_dir() and d.name.startswith("Re")])
print(f"Found {len(re_folders)} Re folders")

# Randomly select 70% of Re folders for training
n_train_folders = int(len(re_folders) * 0.7)
train_folders = random.sample(re_folders, n_train_folders)
test_folders = [f for f in re_folders if f not in train_folders]

print(f"Training on {len(train_folders)} Re folders")
print(f"Testing on {len(test_folders)} Re folders")

def load_data_from_folders(folders):
    """Load data from stat_walls_full_Re*.csv files"""
    all_data = []
    
    for folder in folders:
        # Extract Reynolds number from folder name
        re_value = int(folder.name.replace("Re", ""))
        
        # Find the corresponding CSV file
        csv_file = folder / f"stat_walls_full_Re{re_value}.csv"
        
        if not csv_file.exists():
            print(f"Warning: {csv_file} not found, skipping...")
            continue
        
        # Read CSV file
        try:
            df = pd.read_csv(csv_file, sep=r'\s+')
            
            # Extract relevant columns
            # Columns: cellnumber, y-wall-shear, x-wall-shear
            cellnumber = df['cellnumber'].values
            y_wall_shear = df['y-wall-shear'].values
            x_wall_shear = df['x-wall-shear'].values
            
            # Create Re column
            re_column = np.full(len(df), re_value)
            
            # Combine data
            folder_data = np.column_stack([cellnumber, re_column, x_wall_shear, y_wall_shear])
            all_data.append(folder_data)
            
            print(f"Loaded {len(df)} samples from Re={re_value}")
            
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            continue
    
    # Combine all data
    combined_data = np.vstack(all_data)
    print(f"\nTotal samples loaded: {len(combined_data)}")
    
    return combined_data

# Load training and testing data
print("\n=== Loading Training Data ===")
train_data = load_data_from_folders(train_folders)

print("\n=== Loading Testing Data ===")
test_data = load_data_from_folders(test_folders)

# Split into inputs (X) and outputs (y)
# X: [cellnumber, Re]
# y: [x-wall-shear, y-wall-shear]
X_train = train_data[:, :2]  # cellnumber, Re
y_train = train_data[:, 2:]  # x-wall-shear, y-wall-shear

X_test = test_data[:, :2]
y_test = test_data[:, 2:]

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

# Normalize the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train)

X_test_scaled = scaler_X.transform(X_test)
y_test_scaled = scaler_y.transform(y_test)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train_scaled)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test_scaled)

# Create custom dataset
class WallShearDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create data loaders
train_dataset = WallShearDataset(X_train_tensor, y_train_tensor)
test_dataset = WallShearDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 64)   # Input: cellnumber, Re
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 2)   # Output: x-wall-shear, y-wall-shear
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Initialize model
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\n=== Model Architecture ===")
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")

# Training loop
num_epochs = 50
train_losses = []
test_losses = []

print("\n=== Training ===")
for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    for batch_X, batch_y in train_loader:
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    # Evaluation
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item()
    
    test_loss /= len(test_loader)
    test_losses.append(test_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training History')
plt.legend()
plt.grid(True)
plt.savefig(Path(__file__).parent / 'training_history.png')
print(f"\nTraining history plot saved to: {Path(__file__).parent / 'training_history.png'}")

# Final evaluation
model.eval()
with torch.no_grad():
    train_pred = model(X_train_tensor)
    test_pred = model(X_test_tensor)
    
    # Inverse transform predictions
    train_pred_original = scaler_y.inverse_transform(train_pred.numpy())
    test_pred_original = scaler_y.inverse_transform(test_pred.numpy())
    
    # Calculate R² score
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    
    train_r2 = r2_score(y_train, train_pred_original)
    test_r2 = r2_score(y_test, test_pred_original)
    
    train_mae = mean_absolute_error(y_train, train_pred_original)
    test_mae = mean_absolute_error(y_test, test_pred_original)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred_original))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred_original))
    
    print("\n=== Final Results ===")
    print(f"Training R² Score: {train_r2:.6f}")
    print(f"Testing R² Score: {test_r2:.6f}")
    print(f"\nTraining MAE: {train_mae:.6e}")
    print(f"Testing MAE: {test_mae:.6e}")
    print(f"\nTraining RMSE: {train_rmse:.6e}")
    print(f"Testing RMSE: {test_rmse:.6e}")

# Save the model
model_path = Path(__file__).parent / 'wall_shear_model.pth'
torch.save({
    'model_state_dict': model.state_dict(),
    'scaler_X': scaler_X,
    'scaler_y': scaler_y,
    'train_folders': [f.name for f in train_folders],
    'test_folders': [f.name for f in test_folders]
}, model_path)

print(f"\nModel saved to: {model_path}")

# Example prediction
print("\n=== Example Prediction ===")
example_input = np.array([[0.5, 0.5, 1000]])  # x=0.5, y=0.5, Re=1000
example_input_scaled = scaler_X.transform(example_input)
example_input_tensor = torch.FloatTensor(example_input_scaled)

with torch.no_grad():
    example_pred_scaled = model(example_input_tensor)
    example_pred = scaler_y.inverse_transform(example_pred_scaled.numpy())

print(f"Input: x={example_input[0,0]}, y={example_input[0,1]}, Re={example_input[0,2]}")
print(f"Predicted x-wall-shear: {example_pred[0,0]:.6e}")
print(f"Predicted y-wall-shear: {example_pred[0,1]:.6e}")
