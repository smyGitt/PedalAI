import io
import os
import zipfile
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

class PedalDataset(Dataset):
    def __init__(self, zip_path):
        self.zip_path = zip_path
        with zipfile.ZipFile(zip_path, 'r') as zf:
            self.names = [n for n in zf.namelist() if n.endswith('.pt')]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        # Open zip per-call — required for multiprocessing worker safety
        with zipfile.ZipFile(self.zip_path, 'r') as zf:
            with zf.open(self.names[idx]) as f:
                data = torch.load(io.BytesIO(f.read()), weights_only=True)
        x = data['x'].squeeze(0)  # (T, 128)
        # Compute 12-dim chroma by summing pitch activations across octaves
        chroma = torch.stack([x[:, c::12].sum(dim=1) for c in range(12)], dim=1)  # (T, 12)
        x = torch.cat([x, chroma], dim=1)  # (T, 140)
        return x, data['y'].squeeze()  # (T,) — collapses (1, T, 1) to (T,) for correct BCE alignment

class BiLSTMPedalRegressor(nn.Module):
    def __init__(self, input_size=140, hidden_size=256, num_layers=2):
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
        
        # Regression Head mapping [512] -> [1]
        self.regression_head = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        # Reapply contiguous parameter layout — required every forward pass on CUDA
        self.lstm.flatten_parameters()

        # Force contiguous memory allocation for the input tensor
        x = x.contiguous()

        # x shape: (Batch, sequence_length, 140)
        lstm_out, _ = self.lstm(x)
        
        # lstm_out shape: (Batch, sequence_length, 512)
        pedal_value = self.regression_head(lstm_out)
        
        return pedal_value

def train_bilstm(zip_path, checkpoint_path, epochs=10, save_every_n_files=50, validation_split=0.1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = PedalDataset(zip_path)
    
    # Partition dataset into training and validation subsets
    dataset_size = len(dataset)
    val_size = int(dataset_size * validation_split)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Validation loader — fixed order, created once
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Binary Cross-Entropy loss for binary pedal targets.
    # BCEWithLogitsLoss expects raw logits (no sigmoid in forward) and applies
    # sigmoid internally for numerical stability.
    criterion = nn.BCEWithLogitsLoss()
    
    start_epoch = 0
    file_offset = 0

    # Enforce secure deserialization via weights_only=True
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        # Reapply memory contiguous alignment after state dictionary deserialization
        model.lstm.flatten_parameters()
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        file_offset = checkpoint.get('file_offset', 0)
    
    # TBPTT chunk length — bounds sequence length within cuDNN limits and maximizes GPU utilization
    CHUNK_SIZE = 512

    # AMP GradScaler for FP16 mixed precision on CUDA (no-op on CPU)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    total_train_files = train_size

    for epoch in range(start_epoch, epochs):
        # Epoch-seeded deterministic shuffle — same file order reproduced on resume
        g = torch.Generator()
        g.manual_seed(epoch)
        sampler = torch.utils.data.RandomSampler(train_dataset, generator=g)
        train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=8,         # Maximized asynchronous subprocesses for parallel I/O
            pin_memory=True        # Enabled page-locked memory for high-bandwidth DMA transfers to VRAM
        )

        epoch_offset = file_offset if epoch == start_epoch else 0

        model.train()
        running_loss = 0.0

        train_iter = iter(train_loader)

        # Fast-forward past already-processed files on mid-epoch resume
        if epoch_offset > 0:
            for _ in range(epoch_offset):
                try: next(train_iter)
                except StopIteration: break

        epoch_iterator = tqdm(train_iter, total=total_train_files, initial=epoch_offset, desc=f"Epoch {epoch+1}/{epochs} [Train]", unit="file")

        for file_idx, (x_full, y_full) in enumerate(epoch_iterator, start=epoch_offset):
            # Transfer tensors to NVIDIA VRAM utilizing non-blocking DMA
            x_full = x_full.to(device, non_blocking=True).float()
            y_full = y_full.to(device, non_blocking=True).float()

            T = x_full.shape[1]
            file_loss = 0.0
            num_chunks = 0

            # TBPTT: process each song in fixed-length chunks.
            # Keeps sequences within cuDNN limits and increases gradient update frequency.
            for start in range(0, T, CHUNK_SIZE):
                end = min(start + CHUNK_SIZE, T)
                x_chunk = x_full[:, start:end, :]
                y_chunk = y_full[:, start:end]

                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                    out_chunk = model(x_chunk).squeeze(-1)
                    loss = criterion(out_chunk, y_chunk)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                file_loss += loss.item()
                num_chunks += 1

            avg_file_loss = file_loss / num_chunks if num_chunks > 0 else 0.0
            running_loss += avg_file_loss

            epoch_iterator.set_postfix(BCE_Loss=f"{avg_file_loss:.4f}")

            # Parameterized checkpoint trigger utilizing modulo arithmetic on the enumerable index
            if (file_idx + 1) % save_every_n_files == 0:
                torch.save({
                    'epoch': epoch,
                    'file_offset': file_idx + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_path)

        # Empirical Validation Loop
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            val_iterator = tqdm(val_loader, total=len(val_loader), desc=f"Epoch {epoch+1}/{epochs} [Val]", unit="file")
            for x_val, y_val in val_iterator:
                x_val = x_val.to(device, non_blocking=True).float()
                y_val = y_val.to(device, non_blocking=True).float()

                T_val = x_val.shape[1]
                file_val_loss = 0.0
                num_val_chunks = 0

                for start in range(0, T_val, CHUNK_SIZE):
                    end = min(start + CHUNK_SIZE, T_val)
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                        out_chunk = model(x_val[:, start:end, :]).squeeze(-1)
                        v_loss = criterion(out_chunk, y_val[:, start:end])
                    file_val_loss += v_loss.item()
                    num_val_chunks += 1

                val_loss += file_val_loss / num_val_chunks if num_val_chunks > 0 else 0.0
                
        # Metric Aggregation and Scheduling
        avg_train_loss = running_loss / total_train_files if total_train_files > 0 else 0.0
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        
        print(f"Epoch {epoch+1}/{epochs} | Avg Train BCE Loss: {avg_train_loss:.4f} | Avg Val BCE Loss: {avg_val_loss:.4f}")
        
        # Modulate optimization scalar based on validation convergence
        scheduler.step(avg_val_loss)
        
        # End of Epoch checkpoint serialization
        torch.save({
            'epoch': epoch + 1,
            'file_offset': 0,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)

ZIP_PATH = 'training_tensors.zip'
CHECKPOINT_PATH = 'pedal_bilstm_checkpoint.pt'

if __name__ == '__main__':
    train_bilstm(ZIP_PATH, CHECKPOINT_PATH)