import os
import math
import mido
import torch
from pathlib import Path
from tqdm import tqdm

def process_midi_to_tensor(midi_path: Path, output_dir: str, hz: int = 50) -> bool:
    """
    Parses a raw MIDI binary, applies 50Hz temporal discretization, 
    normalizes velocities/telemetry to FP32 [0.0, 1.0], and serializes to .pt
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
            active_notes[msg.note] = (current_frame, msg.velocity / 127.0)

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
            last_pedal_value = msg.value / 127.0  # Normalize telemetry

    # Post-iteration: Fill the remaining tensor frames to the end of the sequence
    Y[0, last_pedal_frame:total_frames, 0] = last_pedal_value
    
    for note, (start_f, vel) in active_notes.items():
        X[0, start_f:total_frames, note] = vel

    # Serialize to disk
    # POP909 uses duplicate names (001.mid) across subdirectories. 
    # Prefixing with the parent directory name ensures unique tensor filenames.
    parent_id = midi_path.parent.name
    out_filename = f"POP909_{parent_id}_{midi_path.stem}.pt"
    out_path = os.path.join(output_dir, out_filename)
    
    torch.save({'x': X, 'y': Y}, out_path)
    return True

def compile_dataset(source_dir: str, output_dir: str):
    source_path = Path(source_dir)
    
    if not source_path.exists():
        print(f"Error: Source directory '{source_dir}' not found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    midi_files = list(source_path.rglob("*.mid"))
    total_files = len(midi_files)
    
    print(f"Found {total_files} MIDI binaries. Beginning tensor compilation...")
    
    successful_compilations = 0
    
    for midi_file in tqdm(midi_files, desc="Serializing Matrices", unit="file"):
        if process_midi_to_tensor(midi_file, output_dir):
            successful_compilations += 1

    print("\n" + "="*50)
    print("      TENSOR COMPILATION COMPLETE")
    print("="*50)
    print(f"Binaries Parsed      : {total_files}")
    print(f"Tensors Serialized   : {successful_compilations}")
    print(f"Output Directory     : {os.path.abspath(output_dir)}")
    print("="*50)

if __name__ == "__main__":
    # Local relative paths mapping to your Windows 11 working directory
    POP909_DATASET_FOLDER = "POP909"
    OUTPUT_TENSOR_DIR = "pop909_tensors_tbptt"
    
    compile_dataset(POP909_DATASET_FOLDER, OUTPUT_TENSOR_DIR)