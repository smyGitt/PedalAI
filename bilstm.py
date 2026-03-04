import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

class POP909FullSequenceDataset(Dataset):
    def __init__(self, tensor_dir):
        self.tensor_dir = tensor_dir
        self.file_names = [f for f in os.listdir(tensor_dir) if f.endswith('.pt')]
        
    def __len__(self):
        return len(self.file_names)
        
    def __getitem__(self, idx):
        file_path = os.path.join(self.tensor_dir, self.file_names[idx])
        # weights_only=True ensures secure deserialization of compiled tensors
        data = torch.load(file_path, weights_only=True)
        # Squeeze the pre-compiled batch dimension to prevent the DataLoader from creating a 4D tensor
        return data['x'].squeeze(0), data['y'].squeeze(0)

class BiLSTMPedalRegressor(nn.Module):
    def __init__(self, input_size=128, hidden_size=256, num_layers=2):
        super(BiLSTMPedalRegressor, self).__init__()
        
        # bidirectional=True doubles the hidden output dimension
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True
        )
        
        # Force contiguous memory allocation for LSTM weights during initialization
        self.lstm.flatten_parameters()
        
        # Continuous Regression Head mapping [512] -> [1]
        self.regression_head = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        # Force contiguous memory allocation for the input tensor
        x = x.contiguous()
        
        # x shape: (Batch, sequence_length, 128)
        lstm_out, _ = self.lstm(x)
        
        # lstm_out shape: (Batch, sequence_length, 512)
        pedal_value = self.regression_head(lstm_out)
        
        return pedal_value

def train_bilstm(tensor_dir, checkpoint_path, epochs=10, save_every_n_files=50, validation_split=0.1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Hardware Backend Initialized: {device}")
    
    dataset = POP909FullSequenceDataset(tensor_dir)
    
    # Partition dataset into training and validation subsets
    dataset_size = len(dataset)
    val_size = int(dataset_size * validation_split)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # DataLoader configuration for 32GB RAM / RTX 3080 environment
    train_loader = DataLoader(
        train_dataset, 
        batch_size=1,          # Stochastic evaluation using the Adam optimizer
        shuffle=True, 
        num_workers=8,         # Maximized asynchronous subprocesses for parallel I/O
        pin_memory=True        # Enabled page-locked memory for high-bandwidth DMA transfers to VRAM
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    model = BiLSTMPedalRegressor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Implement learning rate scheduling based on empirical validation plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    # L1 Loss (Mean Absolute Error) function
    criterion = nn.L1Loss()
    
    start_epoch = 0
    
    # Enforce secure deserialization via weights_only=True
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        # Reapply memory contiguous alignment after state dictionary deserialization
        model.lstm.flatten_parameters()
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"State dictionary loaded securely. Resuming training from Epoch {start_epoch + 1}")
    
    total_train_files = len(train_loader)
    
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        
        epoch_iterator = tqdm(train_loader, total=total_train_files, desc=f"Epoch {epoch+1}/{epochs} [Train]", unit="file")
        
        for file_idx, (x_full, y_full) in enumerate(epoch_iterator):
            # Transfer tensors to NVIDIA VRAM utilizing non-blocking DMA
            x_full = x_full.to(device, non_blocking=True).float()
            y_full = y_full.to(device, non_blocking=True).float()
            
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass: full dimensional sequence evaluation
            out_reg = model(x_full)
            
            # Execute dimensional reduction to align out_reg (B, T, 1) with y_full (B, T)
            out_reg = out_reg.squeeze(-1)
            
            # Compute Absolute Error across all time steps
            loss = criterion(out_reg, y_full)
            
            # Backward pass
            loss.backward()
            
            # Execute gradient norm clipping to prevent gradient explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Parameter optimization
            optimizer.step()
            
            running_loss += loss.item()
            
            epoch_iterator.set_postfix(L1_Loss=f"{loss.item():.4f}")
            
            # Parameterized checkpoint trigger utilizing modulo arithmetic on the enumerable index
            if (file_idx + 1) % save_every_n_files == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_path)
                tqdm.write(f"Checkpoint serialized at Epoch {epoch+1}, File {file_idx+1}")
                
        # Empirical Validation Loop
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            val_iterator = tqdm(val_loader, total=len(val_loader), desc=f"Epoch {epoch+1}/{epochs} [Val]", unit="file")
            for x_val, y_val in val_iterator:
                x_val = x_val.to(device, non_blocking=True).float()
                y_val = y_val.to(device, non_blocking=True).float()
                
                out_val = model(x_val).squeeze(-1)
                v_loss = criterion(out_val, y_val)
                val_loss += v_loss.item()
                
        # Metric Aggregation and Scheduling
        avg_train_loss = running_loss / total_train_files if total_train_files > 0 else 0.0
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        
        print(f"Epoch {epoch+1}/{epochs} | Avg Train L1 Loss: {avg_train_loss:.4f} | Avg Val L1 Loss: {avg_val_loss:.4f}")
        
        # Modulate optimization scalar based on validation convergence
        scheduler.step(avg_val_loss)
        
        # End of Epoch checkpoint serialization
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)

# Variables mapped to the generated POP909 tensor output directory
TENSOR_DIR = 'pop909_tensors_tbptt'
CHECKPOINT_PATH = 'pop909_bilstm_checkpoint.pt'

if __name__ == '__main__':
    train_bilstm(TENSOR_DIR, CHECKPOINT_PATH)