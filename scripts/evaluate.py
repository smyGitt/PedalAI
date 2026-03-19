import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from bilstm import PedalDataset, BiLSTMPedalRegressor

ZIP_PATH = 'training_tensors.zip'
CHECKPOINT_PATH = 'pedal_bilstm_dropout03_checkpoint.pt'
CHUNK_SIZE = 512
VALIDATION_SPLIT = 0.1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Evaluating on: {device}")

tensor_dir = input("Path to extracted training tensor folder (leave blank to use zip): ").strip()
folder_paths = [tensor_dir] if tensor_dir else None

dataset = PedalDataset(ZIP_PATH, folder_paths=folder_paths)
dataset_size = len(dataset)
val_size = int(dataset_size * VALIDATION_SPLIT)
train_size = dataset_size - val_size
split_gen = torch.Generator().manual_seed(42)
_, val_dataset = random_split(dataset, [train_size, val_size], generator=split_gen)

val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

model = BiLSTMPedalRegressor().to(device)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
all_preds = []
all_labels = []

with torch.no_grad():
    for x, y in tqdm(val_loader, desc="Evaluating", unit="file"):
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).float()

        T = x.shape[1]
        num_chunks = (T + CHUNK_SIZE - 1) // CHUNK_SIZE
        pad_len = num_chunks * CHUNK_SIZE - T

        if pad_len > 0:
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_len))

        x_chunks = x.squeeze(0).view(num_chunks, CHUNK_SIZE, -1)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            out = model(x_chunks).squeeze(-1)

        all_preds.append(torch.sigmoid(out).view(-1)[:T])
        all_labels.append(y.view(-1)[:T])

all_preds = torch.cat(all_preds)
all_labels = torch.cat(all_labels)
total_frames = all_preds.shape[0]

print(f"\nResults ({total_frames:,} frames across {len(val_dataset)} files):\n")
print(f"  {'Threshold':<12} {'Accuracy':>10}")
print(f"  {'-'*12} {'-'*10}")

best_acc = 0
best_thresh = 0
for t in thresholds:
    correct = ((all_preds >= t).float() == all_labels).sum().item()
    acc = correct / total_frames * 100
    if acc > best_acc:
        best_acc = acc
        best_thresh = t
    print(f"  {t:<12} {acc:>9.2f}%")

print(f"\n  Best: {best_thresh} ({best_acc:.2f}%)")
