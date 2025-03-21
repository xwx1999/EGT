import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

class CNN(nn.Module):
    def __init__(self, input_dim=4000, num_channels=64, kernel_size=3):
        super(CNN, self).__init__()
        
        # Reshape input to 2D (batch_size, 1, input_dim)
        self.reshape = lambda x: x.view(x.size(0), 1, -1)
        
        # CNN layers
        self.conv1 = nn.Conv1d(1, num_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(num_channels)
        self.conv2 = nn.Conv1d(num_channels, num_channels*2, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(num_channels*2)
        self.conv3 = nn.Conv1d(num_channels*2, num_channels*4, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn3 = nn.BatchNorm1d(num_channels*4)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(num_channels*4, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 1)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Reshape input
        x = self.reshape(x)
        
        # CNN layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global average pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x.squeeze(-1)
    
    def fit(self, X, y, batch_size=32, epochs=100, learning_rate=0.001, device='cuda'):
        """Train the model"""
        self.to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        # Convert data to tensors
        X = torch.FloatTensor(X).to(device)
        y = torch.FloatTensor(y).to(device)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        start_time = time.time()
        
        # Training loop
        self.train()
        for epoch in range(epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        training_time = time.time() - start_time
        return training_time
    
    def predict(self, X, device='cuda'):
        """Make predictions"""
        self.eval()
        X = torch.FloatTensor(X).to(device)
        
        with torch.no_grad():
            predictions = self(X)
        
        return predictions.cpu().numpy()
    
    def get_params(self):
        """Get model parameters"""
        return {
            'input_dim': self.conv1.in_channels,
            'num_channels': self.conv1.out_channels,
            'kernel_size': self.conv1.kernel_size[0]
        } 