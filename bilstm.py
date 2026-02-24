import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class MaestroFullSequenceDataset(Dataset):
    def __init__(self, tensor_dir):
        self.tensor_dir = tensor_dir
        self.file_names = [f for f in os.listdir(tensor_dir) if f.endswith('.pt')]
        
    def __len__(self):
        return len(self.file_names)
        
    def __getitem__(self, idx):
        file_path = os.path.join(self.tensor_dir, self.file_names[idx])
        data = torch.load(file_path)
        return data['x'], data['y']

class BiLSTMPedalRegressor(nn.Module):
    def __init__(self, input_size=128, hidden_size=256, num_layers=2):
        super(BiLSTMPedalRegressor, self).__init__()
        
        # bidirectional=True doubles the hidden output size
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True
        )
        
        # Pure Continuous Regression Head
        self.regression_head = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        # Force the LSTM's internal weights to be contiguous in memory
        self.lstm.flatten_parameters()
        
        # Force the input tensor to be contiguous in memory
        x = x.contiguous()
        
        # x shape: (1, total_time_steps, 128)
        lstm_out, _ = self.lstm(x)
        
        # lstm_out shape: (1, total_time_steps, 512)
        pedal_value = self.regression_head(lstm_out)
        
        return pedal_value

def train_bilstm(tensor_dir, checkpoint_path, epochs=10, save_every_n_files=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing on: {device}")
    
    # Bypass cuDNN to bypass NVIDIA's hardcoded sequence length limits.
    # This forces PyTorch to use native CUDA kernels, supporting massive un-chunked MIDI files.
    if device.type == 'cuda':
        torch.backends.cudnn.enabled = False
        print("cuDNN disabled: Utilizing native PyTorch CUDA kernels for infinite sequence length support.")
    
    dataset = MaestroFullSequenceDataset(tensor_dir)
    
    # DataLoader optimized for multi-core desktop environment (32GB RAM / RTX 3080)
    dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=True, 
        num_workers=8,       # Spawns 8 asynchronous subprocesses for parallel I/O and deserialization
        pin_memory=True      # Allocates page-locked memory for high-bandwidth DMA transfers to VRAM
    )
    
    model = BiLSTMPedalRegressor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # L1 Loss (Mean Absolute Error) to force sharp transients for pedal mechanical accuracy
    criterion = nn.L1Loss()
    
    start_epoch = 0
    files_processed = 0
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming training from Epoch {start_epoch + 1}")
    
    model.train()
    total_dataset_files = len(dataloader)
    
    for epoch in range(start_epoch, epochs):
        running_loss = 0.0
        
        # Initialize the dynamic CLI progress bar for the current epoch
        epoch_iterator = tqdm(dataloader, total=total_dataset_files, desc=f"Epoch {epoch+1}/{epochs}", unit="file")
        
        for file_idx, (x_full, y_full) in enumerate(epoch_iterator):
            # Enforce strictly contiguous 32-bit floating point tensors
            x_full = x_full.to(device, non_blocking=True).float()
            y_full = y_full.to(device, non_blocking=True).float()
            
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass: Entire sequence processed simultaneously in FP32
            out_reg = model(x_full)
            
            # Compute Absolute Error across all time steps
            loss = criterion(out_reg, y_full)
            
            # Standard FP32 Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            files_processed += 1
            
            # Append the current mathematical loss to the CLI progress bar
            epoch_iterator.set_postfix(L1_Loss=f"{loss.item():.4f}")
            
            if files_processed % save_every_n_files == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_path)
                # Use tqdm.write to output telemetry without breaking the ASCII bar visual
                tqdm.write(f"Checkpoint saved at Epoch {epoch+1}, File {file_idx+1}")
                
        # Calculate Epoch Average Loss based on total files processed in the epoch
        if total_dataset_files > 0:
            avg_loss = running_loss / total_dataset_files
            print(f"Epoch {epoch+1}/{epochs} | Average Sequence L1 Loss: {avg_loss:.4f}")
        
        # End of Epoch Checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)

# Local relative paths mapping to the current working directory
TENSOR_DIR = 'maestro_tensors_tbptt'
CHECKPOINT_PATH = 'maestro_bilstm_checkpoint.pt'

if __name__ == '__main__':
    train_bilstm(TENSOR_DIR, CHECKPOINT_PATH)