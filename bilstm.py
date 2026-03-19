import io
import os
import zipfile
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm


class PedalDataset(Dataset):
    MANIFEST_NAME = 'training_tensors_manifest.txt'

    def __init__(self, zip_path, folder_paths=None):
        """Load .pt files from an extracted folder (preferred) or zip fallback.

        Args:
            zip_path: Path to the zip archive (used as fallback).
            folder_paths: List of candidate folder paths to check for extracted
                          .pt files (e.g. external SSD mount points). The first
                          folder that exists and contains .pt files is used.
        """
        self.folder = None
        self.zip_path = None

        # Try extracted folder — much faster I/O than zip
        for folder in (folder_paths or []):
            manifest = os.path.join(folder, self.MANIFEST_NAME)
            if os.path.isdir(folder) and os.path.exists(manifest):
                with open(manifest, 'r') as f:
                    self.names = [line.strip() for line in f if line.strip()]
                folder_files = set(os.listdir(folder))
                if all(n in folder_files for n in self.names):
                    self.folder = folder
                    print(f"Using extracted folder: {folder} ({len(self.names)} files)")
                    break
                else:
                    missing = [n for n in self.names if n not in folder_files]
                    print(f"[WARNING] Folder {folder} is missing {len(missing)} files, falling back to zip")

        # Fall back to zip
        if self.folder is None:
            self.zip_path = zip_path
            with zipfile.ZipFile(zip_path, 'r') as zf:
                self.names = [os.path.basename(n) for n in zf.namelist() if n.endswith('.pt')]
            print(f"Using zip archive: {zip_path} ({len(self.names)} files)")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if self.folder is not None:
            data = torch.load(os.path.join(self.folder, self.names[idx]), weights_only=True)
        else:
            # Open zip per-call — required for multiprocessing worker safety
            with zipfile.ZipFile(self.zip_path, 'r') as zf:
                with zf.open('training_tensors/' + self.names[idx]) as f:
                    data = torch.load(io.BytesIO(f.read()), weights_only=True)
        x = data['x'].squeeze(0)  # (T, 128)
        # Compute 12-dim chroma by summing pitch activations across octaves
        chroma = torch.stack([x[:, c::12].sum(dim=1) for c in range(12)], dim=1)  # (T, 12)
        x = torch.cat([x, chroma], dim=1)  # (T, 140)
        return x, data['y'].squeeze()  # (T,) — collapses (1, T, 1) to (T,)


class BiLSTMPedalRegressor(nn.Module):
    def __init__(self, input_size=140, hidden_size=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        # Regression head: [hidden*2] -> [1]
        self.regression_head = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        # flatten_parameters() is a CUDA-only memory optimization.
        # Guarding with is_cuda prevents it from being traced by the ONNX exporter,
        # which would produce NaN outputs in the exported graph.
        if x.is_cuda:
            self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(x.contiguous())
        return self.regression_head(self.dropout(lstm_out))


def _save_checkpoint(path, state):
    """Atomic checkpoint save: write to a temp file then rename.

    os.replace() is atomic on both POSIX and Windows NTFS, so a crash
    mid-write leaves the previous checkpoint intact rather than producing
    a partially-written, corrupt file.
    """
    tmp = path + '.tmp'
    torch.save(state, tmp)
    os.replace(tmp, path)


def _has_nan_weights(model):
    """Return list of parameter names that contain non-finite values."""
    return [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]


def train_bilstm(zip_path, checkpoint_path, folder_paths=None, epochs=10, save_every_n_files=50, validation_split=0.1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    # Most chunks are exactly CHUNK_SIZE frames, so cuDNN can cache the
    # optimal LSTM kernel after the first pass rather than re-searching.
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    dataset = PedalDataset(zip_path, folder_paths=folder_paths)
    dataset_size = len(dataset)
    val_size = int(dataset_size * validation_split)
    train_size = dataset_size - val_size
    # Fixed seed ensures the same train/val split across runs, so resumed
    # training never leaks validation files into the training set.
    split_gen = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=split_gen)

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    model = BiLSTMPedalRegressor().to(device)

    # lr=0.0005: halved from 0.001 to reduce risk of divergence in later epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # BCEWithLogitsLoss expects raw logits; sigmoid is applied internally for stability.
    # reduction='none' returns per-element loss, allowing masked averaging for padded chunks.
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    # AMP GradScaler is only meaningful on CUDA. When disabled (CPU), it is a
    # transparent no-op — but critically it does NOT provide NaN/Inf guard on CPU,
    # so we enforce that ourselves with the explicit loss check below.
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    start_epoch = 0
    file_offset = 0

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.lstm.flatten_parameters()
        bad = _has_nan_weights(model)
        if bad:
            raise ValueError(f"Checkpoint contains non-finite weights: {bad}. Cannot resume training.")
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            # Legacy checkpoint without scheduler state — re-apply target lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.0005
        if 'scaler_state_dict' in checkpoint and use_amp:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch']
        file_offset = checkpoint.get('file_offset', 0)
        print(f"Resumed from epoch {start_epoch}, file offset {file_offset}")

    CHUNK_SIZE = 512
    MAX_BATCH_CHUNKS = 200  # ~102,400 frames (~34 min at 50Hz) — OOM safety cap
    # Abort training if this many consecutive NaN events (chunks or file-level
    # grad checks) occur without a successful optimizer step in between.
    MAX_CONSECUTIVE_NAN = 20
    # Abort if this fraction of files in an epoch produce NaN — indicates broad data corruption
    MAX_NAN_RATIO = 0.25

    # Create sampler and DataLoader once; reseed the generator each epoch so
    # persistent_workers stay alive across epochs instead of being respawned.
    train_gen = torch.Generator()
    train_sampler = torch.utils.data.RandomSampler(train_dataset, generator=train_gen)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        sampler=train_sampler,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    for epoch in range(start_epoch, epochs):
        # Reseed the sampler's generator — deterministic shuffle per epoch,
        # same file order reproduced on resume.
        train_gen.manual_seed(epoch)

        # Separate generator with the same seed to pre-compute the permutation
        # for filename lookup without consuming the sampler's generator state.
        sampler_order = torch.randperm(len(train_dataset), generator=torch.Generator().manual_seed(epoch)).tolist()

        epoch_offset = file_offset if epoch == start_epoch else 0
        model.train()
        running_loss = 0.0
        total_files = 0       # total files iterated this epoch
        files_with_loss = 0   # files where at least one finite-loss chunk was processed
        files_with_nan = 0    # files where at least one chunk produced NaN
        consecutive_nan = 0   # tracks run of non-finite losses to detect full divergence
        nan_filenames = []    # track which files produce NaN for diagnosis

        train_iter = iter(train_loader)

        # Fast-forward past already-processed files on mid-epoch resume
        if epoch_offset > 0:
            for _ in range(epoch_offset):
                try:
                    next(train_iter)
                except StopIteration:
                    break

        epoch_iterator = tqdm(
            train_iter,
            total=train_size,
            initial=epoch_offset,
            desc=f"Epoch {epoch + 1}/{epochs} [Train]",
            unit="file",
        )

        for file_idx, (x_full, y_full) in enumerate(epoch_iterator, start=epoch_offset):
            x_full = x_full.to(device, non_blocking=True).float()
            y_full = y_full.to(device, non_blocking=True).float()

            total_files += 1
            # Resolve filename: sampler_order[file_idx] gives the subset index
            # that the sampler yielded at this position, then subset.indices maps
            # that to the original dataset index.
            try:
                original_idx = train_dataset.indices[sampler_order[file_idx]]
                current_filename = dataset.names[original_idx]
            except (IndexError, AttributeError):
                current_filename = f"index_{file_idx}"

            T = x_full.shape[1]
            file_loss = 0.0
            file_had_nan = False
            num_total_chunks = (T + CHUNK_SIZE - 1) // CHUNK_SIZE

            # Pad sequence to exact multiple of CHUNK_SIZE for batched processing
            pad_len = num_total_chunks * CHUNK_SIZE - T
            if pad_len > 0:
                x_full = torch.nn.functional.pad(x_full, (0, 0, 0, pad_len))
                y_full = torch.nn.functional.pad(y_full, (0, pad_len))

            # Reshape (1, T_padded, F) -> (num_chunks, CHUNK_SIZE, F)
            x_chunks = x_full.squeeze(0).view(num_total_chunks, CHUNK_SIZE, -1)
            y_chunks = y_full.squeeze(0).view(num_total_chunks, CHUNK_SIZE)

            optimizer.zero_grad(set_to_none=True)

            if num_total_chunks <= MAX_BATCH_CHUNKS:
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    out_chunks = model(x_chunks).squeeze(-1)
                    per_element_loss = criterion(out_chunks, y_chunks)
                    if pad_len > 0:
                        mask = torch.ones_like(y_chunks)
                        mask[-1, CHUNK_SIZE - pad_len:] = 0.0
                        loss = (per_element_loss * mask).sum() / mask.sum()
                    else:
                        loss = per_element_loss.mean()
            else:
                # Sub-batch to stay within VRAM limits.
                # Each sub-batch gets its own backward() call so activations are
                # freed before the next sub-batch, bounding peak memory.
                total_loss_sum = 0.0
                total_frame_count = 0
                for sb_start in range(0, num_total_chunks, MAX_BATCH_CHUNKS):
                    sb_end = min(sb_start + MAX_BATCH_CHUNKS, num_total_chunks)
                    x_sb = x_chunks[sb_start:sb_end]
                    y_sb = y_chunks[sb_start:sb_end]
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                        out_sb = model(x_sb).squeeze(-1)
                        sb_loss = criterion(out_sb, y_sb)
                        if sb_end == num_total_chunks and pad_len > 0:
                            mask = torch.ones_like(y_sb)
                            mask[-1, CHUNK_SIZE - pad_len:] = 0.0
                            sb_mean = (sb_loss * mask).sum() / mask.sum()
                            sb_count = int(mask.sum().item())
                        else:
                            sb_mean = sb_loss.mean()
                            sb_count = y_sb.numel()
                    if not torch.isfinite(sb_mean):
                        file_had_nan = True
                        break
                    # Weight this sub-batch's gradient by its share of total real frames
                    scaler.scale(sb_mean * (sb_count / T)).backward()
                    total_loss_sum += sb_mean.item() * sb_count
                    total_frame_count += sb_count
                # Compute file-level mean loss for reporting
                if total_frame_count > 0 and not file_had_nan:
                    loss = torch.tensor(total_loss_sum / total_frame_count)
                    file_loss = loss.item()
                else:
                    loss = torch.tensor(float('nan'))

            # ── NaN/Inf guard ────────────────────────────────────────────────
            if not torch.isfinite(loss):
                consecutive_nan += 1
                file_had_nan = True
                tqdm.write(
                    f"[WARNING] Non-finite loss at epoch {epoch + 1}, "
                    f"file '{current_filename}'. "
                    f"Skipping backward. ({consecutive_nan}/{MAX_CONSECUTIVE_NAN})"
                )
                if consecutive_nan >= MAX_CONSECUTIVE_NAN:
                    raise RuntimeError(
                        f"Training aborted: {MAX_CONSECUTIVE_NAN} consecutive "
                        "NaN/Inf losses. The model has diverged — check your "
                        "data quality and consider lowering the learning rate."
                    )
            elif num_total_chunks <= MAX_BATCH_CHUNKS:
                # Standard path: single forward, single backward
                scaler.scale(loss).backward()
                file_loss = loss.item()
            # Sub-batch path: backward already called per sub-batch,
            # file_loss already set above.
            # ────────────────────────────────────────────────────────────────
            num_chunks = num_total_chunks if torch.isfinite(loss) else 0

            # Single optimizer step per file — only if at least one chunk was clean
            if num_chunks > 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # On CPU the AMP scaler is disabled and does NOT inspect gradients —
                # scaler.step() calls optimizer.step() unconditionally. Guard manually
                # so NaN/Inf gradients from a finite loss cannot corrupt weights.
                if not use_amp and not torch.isfinite(grad_norm):
                    consecutive_nan += 1
                    file_had_nan = True
                    tqdm.write(
                        f"[WARNING] Non-finite gradient norm at epoch {epoch + 1}, "
                        f"file '{current_filename}'. "
                        f"Skipping optimizer step. ({consecutive_nan}/{MAX_CONSECUTIVE_NAN})"
                    )
                    if consecutive_nan >= MAX_CONSECUTIVE_NAN:
                        raise RuntimeError(
                            f"Training aborted: {MAX_CONSECUTIVE_NAN} consecutive "
                            "NaN/Inf losses or gradients. The model has diverged — "
                            "check your data quality and consider lowering the learning rate."
                        )
                else:
                    consecutive_nan = 0  # reset only after a fully clean step
                    scaler.step(optimizer)
                    scaler.update()

            if file_had_nan:
                files_with_nan += 1
                nan_filenames.append(current_filename)

            if num_chunks > 0:
                running_loss += file_loss
                files_with_loss += 1
                epoch_iterator.set_postfix(BCE_Loss=f"{file_loss:.4f}")

            if (file_idx + 1) % save_every_n_files == 0:
                bad = _has_nan_weights(model)
                if bad:
                    tqdm.write(f"[ERROR] Non-finite weights before mid-epoch save: {bad}. Skipping save.")
                else:
                    _save_checkpoint(checkpoint_path, {
                        'epoch': epoch,
                        'file_offset': file_idx + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                    })

        # Epoch-level NaN summary
        if files_with_nan > 0:
            nan_ratio = files_with_nan / total_files if total_files > 0 else 0
            print(
                f"[WARNING] Epoch {epoch + 1}: {files_with_nan}/{total_files} files "
                f"produced NaN ({nan_ratio:.1%}). Problem files: {nan_filenames[:10]}"
                + (f" ... and {len(nan_filenames) - 10} more" if len(nan_filenames) > 10 else "")
            )
            if nan_ratio >= MAX_NAN_RATIO:
                raise RuntimeError(
                    f"Training aborted: {nan_ratio:.1%} of files produced NaN this epoch "
                    f"(threshold: {MAX_NAN_RATIO:.0%}). Dataset may be broadly corrupt."
                )

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_files_counted = 0

        with torch.no_grad():
            val_iterator = tqdm(
                val_loader,
                total=len(val_loader),
                desc=f"Epoch {epoch + 1}/{epochs} [Val]",
                unit="file",
            )
            for x_val, y_val in val_iterator:
                x_val = x_val.to(device, non_blocking=True).float()
                y_val = y_val.to(device, non_blocking=True).float()

                T_val = x_val.shape[1]
                num_val_chunks = (T_val + CHUNK_SIZE - 1) // CHUNK_SIZE
                pad_len = num_val_chunks * CHUNK_SIZE - T_val

                if pad_len > 0:
                    x_val = torch.nn.functional.pad(x_val, (0, 0, 0, pad_len))
                    y_val = torch.nn.functional.pad(y_val, (0, pad_len))

                x_vchunks = x_val.squeeze(0).view(num_val_chunks, CHUNK_SIZE, -1)
                y_vchunks = y_val.squeeze(0).view(num_val_chunks, CHUNK_SIZE)

                if num_val_chunks <= MAX_BATCH_CHUNKS:
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                        out_vchunks = model(x_vchunks).squeeze(-1)
                        per_element_loss = criterion(out_vchunks, y_vchunks)
                        if pad_len > 0:
                            mask = torch.ones_like(y_vchunks)
                            mask[-1, CHUNK_SIZE - pad_len:] = 0.0
                            v_loss = (per_element_loss * mask).sum() / mask.sum()
                        else:
                            v_loss = per_element_loss.mean()
                else:
                    total_vloss_sum = 0.0
                    total_vframe_count = 0
                    for sb_start in range(0, num_val_chunks, MAX_BATCH_CHUNKS):
                        sb_end = min(sb_start + MAX_BATCH_CHUNKS, num_val_chunks)
                        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                            out_sb = model(x_vchunks[sb_start:sb_end]).squeeze(-1)
                            sb_loss = criterion(out_sb, y_vchunks[sb_start:sb_end])
                            if sb_end == num_val_chunks and pad_len > 0:
                                mask = torch.ones_like(y_vchunks[sb_start:sb_end])
                                mask[-1, CHUNK_SIZE - pad_len:] = 0.0
                                total_vloss_sum += (sb_loss * mask).sum().item()
                                total_vframe_count += int(mask.sum().item())
                            else:
                                total_vloss_sum += sb_loss.sum().item()
                                total_vframe_count += y_vchunks[sb_start:sb_end].numel()
                    v_loss = torch.tensor(total_vloss_sum / total_vframe_count if total_vframe_count > 0 else float('nan'))

                if torch.isfinite(v_loss):
                    val_loss += v_loss.item()
                    val_files_counted += 1

        # Use files_with_loss (not total train_size) so skipped-NaN files don't
        # deflate the reported average
        avg_train_loss = running_loss / files_with_loss if files_with_loss > 0 else float('nan')
        avg_val_loss = val_loss / val_files_counted if val_files_counted > 0 else float('nan')

        print(f"Epoch {epoch + 1}/{epochs} | Avg Train BCE Loss: {avg_train_loss:.4f} | Avg Val BCE Loss: {avg_val_loss:.4f}")

        # Append to loss history file
        loss_log = os.path.join(os.path.dirname(checkpoint_path) or '.', 'loss_history_dropout03.csv')
        if not os.path.exists(loss_log):
            with open(loss_log, 'w') as f:
                f.write('epoch,train_loss,val_loss\n')
        with open(loss_log, 'a') as f:
            f.write(f'{epoch + 1},{avg_train_loss:.4f},{avg_val_loss:.4f}\n')

        if torch.isfinite(torch.tensor(avg_val_loss)):
            scheduler.step(avg_val_loss)
        else:
            print(f"[WARNING] Skipping scheduler step — avg_val_loss is {avg_val_loss}")

        # Validate weights before end-of-epoch save — never write a corrupt checkpoint
        bad = _has_nan_weights(model)
        if bad:
            raise RuntimeError(
                f"Training aborted: non-finite weights at end of epoch {epoch + 1}: {bad}. "
                "Continuing with a corrupted model is pointless."
            )
        _save_checkpoint(checkpoint_path, {
            'epoch': epoch + 1,
            'file_offset': 0,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
        })


ZIP_PATH = 'training_tensors.zip'
CHECKPOINT_PATH = 'pedal_bilstm_dropout03_checkpoint.pt'

if __name__ == '__main__':
    tensor_dir = input("Path to extracted training tensor folder (leave blank to use zip): ").strip()
    folder_paths = [tensor_dir] if tensor_dir else None
    train_bilstm(ZIP_PATH, CHECKPOINT_PATH, folder_paths=folder_paths)
