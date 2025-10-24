import os
import pandas as pd
import argparse

def split_traces(input_csv, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_csv)

    first_half_rows = []
    second_half_rows = []

    # Group by trace_id
    for trace_id, trace_df in df.groupby("trace_id"):
        max_step = trace_df["step"].max()
        half_step = int(max_step / 2)

        # First half: steps <= half_step
        first_half = trace_df[trace_df["step"] <= half_step].copy()
        first_half["trace_id"] = f"{trace_id}"
        first_half_rows.append(first_half)

        # Second half: steps >= half_step
        second_half = trace_df[trace_df["step"] >= half_step].copy()
        second_half["trace_id"] = f"{trace_id}"
        second_half_rows.append(second_half)

    # Concatenate all halves
    first_half_df = pd.concat(first_half_rows, ignore_index=True)
    second_half_df = pd.concat(second_half_rows, ignore_index=True)

    # Save two separate CSVs
    base_name = os.path.splitext(os.path.basename(input_csv))[0]
    first_half_path = os.path.join(output_dir, f"{base_name}_half0.csv")
    second_half_path = os.path.join(output_dir, f"{base_name}_half1.csv")

    first_half_df.to_csv(first_half_path, index=False)
    second_half_df.to_csv(second_half_path, index=False)

    print(f"Saved first halves to {first_half_path}")
    print(f"Saved second halves to {second_half_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split traces in a CSV into two halves per trace_id based on steps")
    parser.add_argument("input_csv", type=str, help="Path to the input traces CSV")
    parser.add_argument("output_dir", type=str, help="Directory to save the split CSVs")
    args = parser.parse_args()

    split_traces(args.input_csv, args.output_dir)
