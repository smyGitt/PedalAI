import mido
import statistics
from pathlib import Path

def calculate_dataset_statistics(source_dir: str):
    """
    Parses a dataset directory tree and calculates statistical variance 
    metrics for CC64 pedal actuations.
    """
    source_path = Path(source_dir)
    
    if not source_path.exists():
        print(f"Error: Directory '{source_dir}' not found.")
        return
        
    print(f"Scanning '{source_path.absolute()}' for CC64 telemetry...")
    
    actuation_counts = []
    corrupted_files = 0
    total_files = 0

    # Dynamically traverse the target directory tree
    for midi_file in source_path.rglob("*.mid"):
        total_files += 1
        
        try:
            # Load binary with clipping enabled to handle malformed velocity/pitch bytes safely
            mid = mido.MidiFile(str(midi_file), clip=True)
            
            file_edges = 0
            for track in mid.tracks:
                for msg in track:
                    if msg.type == 'control_change' and msg.control == 64:
                        file_edges += 1
                        
            actuation_counts.append(file_edges)
            
        except Exception:
            corrupted_files += 1
            
    # Statistical computation
    if not actuation_counts:
        print("Dataset Evaluation Failed: No valid MIDI binaries parsed.")
        return

    valid_files = len(actuation_counts)
    total_edges = sum(actuation_counts)
    
    mean_actuations = statistics.mean(actuation_counts)
    median_actuations = statistics.median(actuation_counts)
    max_actuations = max(actuation_counts)
    min_actuations = min(actuation_counts)

    # Output standard analytics payload
    print("\n" + "="*40)
    print("      DATASET CC64 EDGE STATISTICS")
    print("="*40)
    print(f"Total Binaries Scanned : {total_files}")
    print(f"Valid Parsed Binaries  : {valid_files}")
    print(f"Corrupted/Unreadable   : {corrupted_files}")
    print("-" * 40)
    print(f"Total Dataset Edges    : {total_edges:,}")
    print(f"Minimum Actuations     : {min_actuations}")
    print(f"Maximum Actuations     : {max_actuations:,}")
    print(f"Mean (Average)         : {mean_actuations:.2f}")
    print(f"Median                 : {median_actuations:.1f}")
    print("="*40)

if __name__ == "__main__":
    # Point this relative path to your extracted POP909 directory
    POP909_DATASET_FOLDER = "POP909"
    
    calculate_dataset_statistics(POP909_DATASET_FOLDER)