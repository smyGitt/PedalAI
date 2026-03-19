import mido
from pathlib import Path

def evaluate_pedal_viability(filepath: str, min_actuations: int = 15) -> bool:
    """
    Evaluates a MIDI file for high-quality, training-viable CC64 pedal data.
    """
    try:
        # Load MIDI without loading external tracks to minimize memory footprint
        mid = mido.MidiFile(filepath, clip=True)
    except Exception:
        # Silently reject corrupted or malformed MIDI binaries
        return False

    pedal_down_count = 0
    pedal_up_count = 0
    
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'control_change' and msg.control == 64:
                if msg.value >= 64:
                    pedal_down_count += 1
                else:
                    pedal_up_count += 1

    total_actuations = pedal_down_count + pedal_up_count

    # Rejection Criteria 1: Insufficient data density
    if total_actuations < min_actuations:
        return False
        
    # Rejection Criteria 2: Static Hold (mechanically invalid state)
    if pedal_down_count > 0 and pedal_up_count == 0:
        return False
        
    # Rejection Criteria 3: Inverse Static Hold
    if pedal_up_count > 0 and pedal_down_count == 0:
        return False

    return True

def scan_dataset(source_dir: str):
    """
    Executes a silent evaluation pass strictly targeting 1-level deep subdirectories.
    """
    source_path = Path(source_dir)
    
    if not source_path.exists():
        print(f"Error: Directory {source_dir} not found.")
        return
        
    total_files = 0
    acceptable_files = 0
    
    # Use rglob("*.mid") to dynamically drill down into any folder depth
    for midi_file in source_path.rglob("*.mid"):
        total_files += 1

        if evaluate_pedal_viability(str(midi_file), 100):
            acceptable_files += 1
            
    # Output exact requested telemetry format
    print(f"{acceptable_files} / {total_files} Acceptable")

if __name__ == "__main__":
    # Point this to your relative POP909 directory located next to the script
    POP909_DATASET_FOLDER = "POP909"
    
    scan_dataset(POP909_DATASET_FOLDER)