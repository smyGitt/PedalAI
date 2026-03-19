import os
import re
import math
import mido
import torch
from pathlib import Path
from tqdm import tqdm


def _sanitize(name: str) -> str:
    """Replaces any character that is not alphanumeric or underscore with an
    underscore, then collapses consecutive underscores to one.
    Needed for GiantMIDI-Piano filenames which contain spaces, commas, and
    other special characters."""
    return re.sub(r'_+', '_', re.sub(r'[^a-zA-Z0-9]', '_', name)).strip('_')


def process_midi_to_tensor(midi_path: Path, output_dir: str, dataset_prefix: str, hz: int = 50) -> bool:
    """
    Parses a raw MIDI binary, applies 50Hz temporal discretization,
    normalizes velocities/telemetry to FP32 [0.0, 1.0], and serializes to .pt

    All tracks are merged into a single piano roll (X) to capture the full
    musical texture the pedal was authored against. This also keeps the
    training distribution consistent with midiTOkey inference, where the
    user selects all tracks of the loaded MIDI.

    dataset_prefix distinguishes tensors by source dataset (e.g. 'POP909', 'GIANTMIDI').
    """
    try:
        # Load MIDI; mido automatically merges tracks when iterated directly
        mid = mido.MidiFile(str(midi_path), clip=True)
    except Exception:
        return False

    # Calculate total required sequence length
    total_time = mid.length
    total_frames = math.ceil(total_time * hz)

    if total_frames <= 0:
        return False

    # Initialize FP32 tensors (Batch=1, Sequence_Length, Features)
    X = torch.zeros((1, total_frames, 128), dtype=torch.float32)
    Y = torch.zeros((1, total_frames, 1), dtype=torch.float32)

    absolute_time = 0.0

    # State tracking dictionaries
    active_notes = {}  # pitch -> (start_frame, normalized_velocity)
    last_pedal_frame = 0
    last_pedal_value = 0.0

    # mido yields messages in strict chronological order with delta times (in seconds)
    for msg in mid:
        absolute_time += msg.time
        current_frame = min(int(absolute_time * hz), total_frames - 1)

        if msg.type == 'note_on' and msg.velocity > 0:
            # Handle repeated note_on without note_off (monophonic re-trigger)
            if msg.note in active_notes:
                start_f, vel = active_notes[msg.note]
                X[0, start_f:current_frame, msg.note] = vel

            # Normalize velocity immediately
            active_notes[msg.note] = (current_frame, 1.0)

        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            if msg.note in active_notes:
                start_f, vel = active_notes[msg.note]
                X[0, start_f:current_frame, msg.note] = vel
                del active_notes[msg.note]

        elif msg.type == 'control_change' and msg.control == 64:
            # Fill the previous temporal interval with the previous state
            Y[0, last_pedal_frame:current_frame, 0] = last_pedal_value

            # Update state trackers for the next interval
            last_pedal_frame = current_frame
            last_pedal_value = 1.0 if msg.value >= 64 else 0.0

    # Post-iteration: Fill the remaining tensor frames to the end of the sequence
    Y[0, last_pedal_frame:total_frames, 0] = last_pedal_value

    for note, (start_f, vel) in active_notes.items():
        X[0, start_f:total_frames, note] = vel

    # Build a unique output filename.
    # POP909 uses duplicate stems (001.mid) across subdirectories — include parent.
    # GiantMIDI uses unique stems but may contain special characters — sanitize both.
    parent_id = _sanitize(midi_path.parent.name)
    stem_id = _sanitize(midi_path.stem)
    out_filename = f"{dataset_prefix}_{parent_id}_{stem_id}.pt"
    out_path = os.path.join(output_dir, out_filename)

    torch.save({'x': X, 'y': Y}, out_path)
    return True


def compile_dataset(source_dir: str, output_dir: str, dataset_prefix: str):
    source_path = Path(source_dir)

    if not source_path.exists():
        print(f"Error: Source directory '{source_dir}' not found.")
        return

    os.makedirs(output_dir, exist_ok=True)

    midi_files = list(source_path.rglob("*.mid"))
    total_files = len(midi_files)

    print(f"[{dataset_prefix}] Found {total_files} MIDI binaries. Beginning tensor compilation...")

    successful_compilations = 0

    for midi_file in tqdm(midi_files, desc=f"[{dataset_prefix}] Serializing", unit="file"):
        if process_midi_to_tensor(midi_file, output_dir, dataset_prefix):
            successful_compilations += 1

    print("\n" + "="*50)
    print(f"      [{dataset_prefix}] TENSOR COMPILATION COMPLETE")
    print("="*50)
    print(f"Binaries Parsed      : {total_files}")
    print(f"Tensors Serialized   : {successful_compilations}")
    print(f"Output Directory     : {os.path.abspath(output_dir)}")
    print("="*50 + "\n")


if __name__ == "__main__":
    # Shared output directory for all datasets.
    # Both POP909 and GiantMIDI tensors are written here and mixed automatically
    # by the DataLoader during training.
    OUTPUT_TENSOR_DIR = r"C:\Users\kevin\Documents\VSCODE\PedalAI\training_tensors_new"

    # --- POP909 ---
    POP909_FOLDER = "POP909"
    compile_dataset(POP909_FOLDER, OUTPUT_TENSOR_DIR, dataset_prefix="POP909")

    # --- GiantMIDI-Piano ---
    # Extract surname_checked_midis_v1.2.zip and point this to the extracted folder.
    # The zip extracts to a single flat directory — all .mid files are in one folder.
    # Example: if extracted to C:\Users\kevin\Downloads\surname_checked_midis, set:
    #   GIANTMIDI_FOLDER = r"C:\Users\kevin\Downloads\surname_checked_midis"
    GIANTMIDI_FOLDER = "surname_checked_midis"
    compile_dataset(GIANTMIDI_FOLDER, OUTPUT_TENSOR_DIR, dataset_prefix="GIANTMIDI")

    # Generate manifest matching zip ordering for DataLoader consistency
    import zipfile
    ZIP_PATH = r'F:\training_tensors.zip'
    if os.path.exists(ZIP_PATH):
        with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
            zip_names = [os.path.basename(n) for n in zf.namelist() if n.endswith('.pt')]
        manifest_path = os.path.join(OUTPUT_TENSOR_DIR, 'training_tensors_manifest.txt')
        with open(manifest_path, 'w') as f:
            for name in zip_names:
                f.write(name + '\n')
        print(f"Manifest written: {manifest_path} ({len(zip_names)} entries)")
    else:
        print(f"WARNING: Zip not found at {ZIP_PATH} — manifest not generated.")
