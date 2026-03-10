import mido
import statistics
from collections import Counter
from pathlib import Path


def classify_cc64_messages(source_dir: str):
    """
    Classifies every CC64 message in the dataset into four categories to
    determine whether the eval2 / eval3 gap is caused by:

      A) True Press       — 0 -> x  (pedal was off, new non-zero value)
      B) True Release     — x -> 0  (pedal was on, now zero)
      C) Value Shift      — x -> y  (pedal held, value changed, both non-zero)
                           This is actual feathering / graduated pedal.
      D) Redundant Zero   — 0 -> 0  (pedal already off, another zero sent)

    Also reports:
      - Distribution of all unique CC64 values observed
      - Count of files that are strictly binary (only values 0 and 127)
      - Count of files that contain no CC64 data at all
    """
    source_path = Path(source_dir)

    if not source_path.exists():
        print(f"Error: Directory '{source_dir}' not found.")
        return

    print(f"Scanning '{source_path.absolute()}' for CC64 classification...\n")

    total_files = 0
    corrupted_files = 0
    files_with_no_pedal = 0
    files_strictly_binary = 0
    files_with_feathering = 0

    # Message-level counters
    true_presses = 0       # 0 -> x
    true_releases = 0      # x -> 0
    value_shifts = 0       # x -> y (feathering)
    redundant_zeros = 0    # 0 -> 0

    # Global value distribution
    value_counter = Counter()

    for midi_file in source_path.rglob("*.mid"):
        total_files += 1

        try:
            mid = mido.MidiFile(str(midi_file), clip=True)
        except Exception:
            corrupted_files += 1
            continue

        file_values = set()
        file_true_presses = 0
        file_value_shifts = 0

        for track in mid.tracks:
            current_value = 0  # CC64 state machine per track

            for msg in track:
                if msg.type != 'control_change' or msg.control != 64:
                    continue

                v = msg.value
                value_counter[v] += 1
                file_values.add(v)

                if current_value == 0 and v > 0:
                    true_presses += 1
                    file_true_presses += 1
                elif current_value > 0 and v == 0:
                    true_releases += 1
                elif current_value > 0 and v > 0:
                    value_shifts += 1
                    file_value_shifts += 1
                else:  # current_value == 0 and v == 0
                    redundant_zeros += 1

                current_value = v

        if not file_values:
            files_with_no_pedal += 1
        else:
            non_zero_values = file_values - {0}
            if non_zero_values.issubset({127}):
                files_strictly_binary += 1
            else:
                files_with_feathering += 1

    total_messages = true_presses + true_releases + value_shifts + redundant_zeros
    valid_files = total_files - corrupted_files

    # Top 20 most common CC64 values
    top_values = value_counter.most_common(20)

    # Count how many distinct non-zero values exist
    non_zero_values_seen = {v: c for v, c in value_counter.items() if v > 0}
    strictly_binary_messages = value_counter[0] + value_counter[127]
    non_binary_messages = sum(c for v, c in value_counter.items() if v not in (0, 127))

    print("=" * 55)
    print("       CC64 MESSAGE CLASSIFICATION REPORT")
    print("=" * 55)
    print(f"Total Binaries Scanned     : {total_files}")
    print(f"Corrupted/Unreadable       : {corrupted_files}")
    print(f"Valid Files                : {valid_files}")
    print()
    print("-" * 55)
    print("  MESSAGE CATEGORIES")
    print("-" * 55)
    print(f"Total CC64 Messages        : {total_messages:,}")
    print(f"  A) True Press  (0 -> x)  : {true_presses:,}  ({100*true_presses/total_messages:.1f}%)")
    print(f"  B) True Release (x -> 0) : {true_releases:,}  ({100*true_releases/total_messages:.1f}%)")
    print(f"  C) Value Shift (x -> y)  : {value_shifts:,}  ({100*value_shifts/total_messages:.1f}%)  <- feathering indicator")
    print(f"  D) Redundant Zero (0->0) : {redundant_zeros:,}  ({100*redundant_zeros/total_messages:.1f}%)")
    print()
    print("-" * 55)
    print("  FILE-LEVEL PEDAL PROFILE")
    print("-" * 55)
    print(f"Files with no CC64 data    : {files_with_no_pedal}")
    print(f"Files strictly binary      : {files_strictly_binary}  (only values 0 and 127)")
    print(f"Files with feathering      : {files_with_feathering}  (non-zero values other than 127)")
    print()
    print("-" * 55)
    print("  VALUE DISTRIBUTION")
    print("-" * 55)
    print(f"Distinct CC64 values seen  : {len(value_counter)}")
    print(f"Messages with value 0      : {value_counter[0]:,}")
    print(f"Messages with value 127    : {value_counter[127]:,}")
    print(f"Messages strictly binary   : {strictly_binary_messages:,}  ({100*strictly_binary_messages/total_messages:.1f}%)")
    print(f"Messages non-binary        : {non_binary_messages:,}  ({100*non_binary_messages/total_messages:.1f}%)")
    print()
    print("  Top 20 most frequent CC64 values:")
    for val, count in top_values:
        bar = "#" * int(40 * count / top_values[0][1])
        print(f"    {val:>3} : {count:>8,}  {bar}")
    print("=" * 55)


if __name__ == "__main__":
    POP909_DATASET_FOLDER = "POP909"
    classify_cc64_messages(POP909_DATASET_FOLDER)
