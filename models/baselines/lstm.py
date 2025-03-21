import torch
import torch.nn as nn
import numpy as np
import time

class LSTM(nn.Module):
    def __init__(self, input_dim=4000, hidden_dim=256, num_layers=2, dropout=0.5):
        super(LSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim * 2, 128)  # *2 for bidirectional
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(64, 1)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Reshape input to (batch_size, 1, input_dim)
        x = x.unsqueeze(1)
        
        # LSTM layers
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  # Take the last output
        
        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
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
            'input_dim': self.lstm.input_size,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers
        } 