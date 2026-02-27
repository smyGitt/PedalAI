import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class POP909FullSequenceDataset(Dataset):
    def __init__(self, tensor_dir):
        self.tensor_dir = tensor_dir
        self.file_names = [f for f in os.listdir(tensor_dir) if f.endswith('.pt')]
        
    def __len__(self):
        return len(self.file_names)
        
    def __getitem__(self, idx):
        file_path = os.path.join(self.tensor_dir, self.file_names[idx])
        # weights_only=True ensures safe deserialization of compiled tensors
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
        
        # Continuous Regression Head mapping [512] -> [1]
        self.regression_head = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        # Force contiguous memory allocation for LSTM weights
        self.lstm.flatten_parameters()
        
        # Force contiguous memory allocation for the input tensor
        x = x.contiguous()
        
        # x shape: (1, sequence_length, 128)
        lstm_out, _ = self.lstm(x)
        
        # lstm_out shape: (1, sequence_length, 512)
        pedal_value = self.regression_head(lstm_out)
        
        return pedal_value

def train_bilstm(tensor_dir, checkpoint_path, epochs=10, save_every_n_files=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Hardware Backend Initialized: {device}")
    
    # Bypass cuDNN to bypass NVIDIA's hardcoded sequence length limits.
    # This forces PyTorch to use native CUDA kernels, supporting massive un-chunked MIDI files.
    if device.type == 'cuda':
        torch.backends.cudnn.enabled = False
        print("cuDNN disabled: Utilizing native PyTorch CUDA kernels for infinite sequence length support.")
    
    dataset = POP909FullSequenceDataset(tensor_dir)
    
    # DataLoader optimized for 32GB RAM / RTX 3080 environment
    dataloader = DataLoader(
        dataset, 
        batch_size=1,          # Stochastic Gradient Descent (Full Sequence)
        shuffle=True, 
        num_workers=8,         # Maximized asynchronous subprocesses for parallel I/O
        pin_memory=True        # Enabled page-locked memory for high-bandwidth DMA transfers to VRAM
    )
    
    model = BiLSTMPedalRegressor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # L1 Loss (Mean Absolute Error) function
    criterion = nn.L1Loss()
    
    start_epoch = 0
    files_processed = 0
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"State dictionary loaded. Resuming training from Epoch {start_epoch + 1}")
    
    model.train()
    total_dataset_files = len(dataloader)
    
    for epoch in range(start_epoch, epochs):
        running_loss = 0.0
        
        epoch_iterator = tqdm(dataloader, total=total_dataset_files, desc=f"Epoch {epoch+1}/{epochs}", unit="file")
        
        for file_idx, (x_full, y_full) in enumerate(epoch_iterator):
            # Transfer tensors to NVIDIA VRAM utilizing non-blocking DMA
            x_full = x_full.to(device, non_blocking=True).float()
            y_full = y_full.to(device, non_blocking=True).float()
            
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass: full dimensional sequence evaluation
            out_reg = model(x_full)
            
            # Compute Absolute Error across all time steps
            loss = criterion(out_reg, y_full)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            files_processed += 1
            
            epoch_iterator.set_postfix(L1_Loss=f"{loss.item():.4f}")
            
            if files_processed % save_every_n_files == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_path)
                tqdm.write(f"Checkpoint serialized at Epoch {epoch+1}, File {file_idx+1}")
                
        if total_dataset_files > 0:
            avg_loss = running_loss / total_dataset_files
            print(f"Epoch {epoch+1}/{epochs} | Average Sequence L1 Loss: {avg_loss:.4f}")
        
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