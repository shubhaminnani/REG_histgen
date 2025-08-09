import pandas as pd
import json
import argparse
import os


def csv_to_corrected_json(csv_path: str, output_json_path: str):
    # Load CSV
    df = pd.read_csv(csv_path)

    predictions = []
    modified_count = 0

    for _, row in df.iterrows():
        report = row["Generated Reports"]
        corrected_report = report.replace(';', ';\n ').replace('<unk> ', '')

        if report != corrected_report:
            modified_count += 1

        predictions.append({
            "id": row["Case ID"] + '.tiff',
            "report": corrected_report
        })

    # Save corrected JSON
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2)

    print(f"Processed {len(predictions)} records.")
    print(f"Modified {modified_count} reports.")
    print(f"Corrected JSON saved to '{output_json_path}'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert CSV to corrected JSON with newline after semicolon.")
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the input CSV file')
    args = parser.parse_args()

    # Output path in same directory as CSV
    base_dir = os.path.dirname(args.csv_path)
    output_json_path = os.path.join(base_dir, 'predictions.json')

    # Convert and save directly as corrected JSON
    csv_to_corrected_json(args.csv_path, output_json_path)
