import mido
import statistics
from pathlib import Path

def calculate_linear_actuations(source_dir: str):
    """
    Parses a dataset directory tree and calculates statistical variance 
    for CC64 actuations strictly defined by absolute zero-origin departures.
    """
    source_path = Path(source_dir)
    
    if not source_path.exists():
        print(f"Error: Directory '{source_dir}' not found.")
        return
        
    print(f"Scanning '{source_path.absolute()}' for zero-origin linear actuations...")
    
    actuation_counts = []
    corrupted_files = 0
    total_files = 0

    for midi_file in source_path.rglob("*.mid"):
        total_files += 1
        
        try:
            mid = mido.MidiFile(str(midi_file), clip=True)
            
            file_actuations = 0
            pedal_is_active = False
            
            for track in mid.tracks:
                for msg in track:
                    if msg.type == 'control_change' and msg.control == 64:
                        
                        # Linear Origin Analysis Logic
                        # 0 -> x (where x > 0)
                        if msg.value > 0 and not pedal_is_active:
                            pedal_is_active = True
                            file_actuations += 1
                            
                        # x -> 0 (where x > 0)
                        elif msg.value == 0 and pedal_is_active:
                            pedal_is_active = False
                            file_actuations += 1
                            
            actuation_counts.append(file_actuations)
            
        except Exception:
            corrupted_files += 1
            
    if not actuation_counts:
        print("Dataset Evaluation Failed: No valid MIDI binaries parsed.")
        return

    valid_files = len(actuation_counts)
    total_actuations = sum(actuation_counts)
    mean_actuations = statistics.mean(actuation_counts)
    median_actuations = statistics.median(actuation_counts)
    max_actuations = max(actuation_counts)
    min_actuations = min(actuation_counts)

    print("\n" + "="*50)
    print("      DATASET LINEAR ACTUATION STATISTICS")
    print("="*50)
    print(f"Total Binaries Scanned   : {total_files}")
    print(f"Valid Parsed Binaries    : {valid_files}")
    print(f"Corrupted/Unreadable     : {corrupted_files}")
    print("\n" + "-" * 50)
    print("  ABSOLUTE MECHANICAL EDGES (0 <-> x)")
    print("-" * 50)
    print(f"Total Dataset Actuations : {total_actuations:,}")
    print(f"Minimum Actuations/File  : {min_actuations:,}")
    print(f"Maximum Actuations/File  : {max_actuations:,}")
    print(f"Mean (Average)           : {mean_actuations:.2f}")
    print(f"Median                   : {median_actuations:.1f}")
    print("="*50)

if __name__ == "__main__":
    POP909_DATASET_FOLDER = "POP909"
    calculate_linear_actuations(POP909_DATASET_FOLDER)