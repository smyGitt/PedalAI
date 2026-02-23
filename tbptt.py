import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class MaestroTBPTTDataset(Dataset):
    def __init__(self, tensor_dir):
        self.tensor_dir = tensor_dir
        self.file_names = [f for f in os.listdir(tensor_dir) if f.endswith('.pt')]
        
    def __len__(self):
        return len(self.file_names)
        
    def __getitem__(self, idx):
        file_path = os.path.join(self.tensor_dir, self.file_names[idx])
        data = torch.load(file_path)
        return data['x'], data['y']

class SustainPedalPredictor(nn.Module):
    def __init__(self, input_size=128, hidden_size=256, num_classes=2, num_layers=2):
        super(SustainPedalPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.classification_head = nn.Linear(hidden_size, num_classes)
        self.regression_head = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden_states=None):
        lstm_out, hidden_states = self.lstm(x, hidden_states)
        class_logits = self.classification_head(lstm_out)
        pedal_value = self.regression_head(lstm_out)
        return class_logits, pedal_value, hidden_states

def train_tbptt(tensor_dir, checkpoint_path, epochs=10, chunk_size=1000, save_every_n_files=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing on: {device}")
    
    dataset = MaestroTBPTTDataset(tensor_dir)
    
    # DataLoader Optimizations applied here
    dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True
    )
    
    model = SustainPedalPredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # AMP Scaler Initialization (Safeguarded for CPU fallback)
    use_cuda = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda)
    
    criterion_classification = nn.CrossEntropyLoss()
    criterion_regression = nn.MSELoss()
    weight_classification = 1.0
    weight_regression = 0.005
    
    start_epoch = 0
    files_processed = 0
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Only load scaler state if resuming on a CUDA-enabled machine
        if 'scaler_state_dict' in checkpoint and use_cuda:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        start_epoch = checkpoint['epoch']
        print(f"Resuming training from Epoch {start_epoch + 1}")
    
    model.train()
    
    for epoch in range(start_epoch, epochs):
        running_loss = 0.0
        total_epoch_chunks = 0  
        
        for file_idx, (x_full, y_full) in enumerate(dataloader):
            x_full = x_full.to(device, non_blocking=True)
            y_full = y_full.to(device, non_blocking=True)
            
            total_time_steps = x_full.size(1)
            hidden_states = None
            
            for t in range(0, total_time_steps, chunk_size):
                x_chunk = x_full[:, t : t + chunk_size, :]
                y_chunk = y_full[:, t : t + chunk_size, :]
                
                if x_chunk.size(1) < chunk_size:
                    break
                    
                optimizer.zero_grad(set_to_none=True)
                
                # Dynamic AMP Autocast Context Manager
                with torch.autocast(device_type=device.type, dtype=torch.float16 if use_cuda else torch.bfloat16, enabled=use_cuda):
                    out_class, out_reg, hidden_states = model(x_chunk, hidden_states)
                    y_chunk_class = (y_chunk > 0.5).long().squeeze(-1) 
                    
                    loss_c = criterion_classification(out_class.transpose(1, 2), y_chunk_class)
                    loss_r = criterion_regression(out_reg, y_chunk)
                    loss_total = (weight_classification * loss_c) + (weight_regression * loss_r)
                
                hidden_states = tuple(h.detach() for h in hidden_states)
                
                scaler.scale(loss_total).backward()
                scaler.step(optimizer)
                scaler.update()
                
                running_loss += loss_total.item()
                total_epoch_chunks += 1
                
            files_processed += 1
            
            if files_processed % save_every_n_files == 0:
                save_dict = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                if use_cuda:
                    save_dict['scaler_state_dict'] = scaler.state_dict()
                    
                torch.save(save_dict, checkpoint_path)
                print(f"Checkpoint saved at Epoch {epoch+1}, File {file_idx+1}")
                
        if total_epoch_chunks > 0:
            avg_loss = running_loss / total_epoch_chunks
            print(f"Epoch {epoch+1}/{epochs} | Average Chunk Loss: {avg_loss}")
        
        save_dict = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        if use_cuda:
            save_dict['scaler_state_dict'] = scaler.state_dict()
            
        torch.save(save_dict, checkpoint_path)

# Local relative paths mapping to the current working directory
TENSOR_DIR = 'maestro_tensors_tbptt'
CHECKPOINT_PATH = 'maestro_tbptt_checkpoint.pt'

# Strictly required entry point safeguard for num_workers multiprocessing on Windows
if __name__ == '__main__':
    train_tbptt(TENSOR_DIR, CHECKPOINT_PATH)