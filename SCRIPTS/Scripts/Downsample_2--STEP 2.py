import os
import pandas as pd
import numpy as np
import re

# === INPUT and OUTPUT folders ===
input_folder = r"C:\TruckIDMe Project\CAN_Data"               
output_folder = r"C:\TruckIDMe Project\Downsampled_data"       
os.makedirs(output_folder, exist_ok=True)


def is_valid_signal(col_name):
    return not (col_name.lower().startswith("id") or "unnamed" in col_name.lower())

def extract_id(col):
    match = re.findall(r'\d+', str(col))
    return match[0] if match else None

for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):
        input_path = os.path.join(input_folder, filename)
        print(f"\n📥 Processing: {filename}")

        try:
            df = pd.read_csv(input_path)
            columns = df.columns.tolist()
            signal_agg_list = []

            for i in range(0, len(columns), 2):
                time_col = columns[i]
                signal_col = columns[i + 1] if i + 1 < len(columns) else None

                if not time_col or not signal_col:
                    continue
                if not is_valid_signal(signal_col) or not is_valid_signal(time_col):
                    continue

                if extract_id(time_col) != extract_id(signal_col):
                    print(f"⛔ Skipping mismatched pair: {time_col} and {signal_col}")
                    continue

                try:
                    sub_df = df[[time_col, signal_col]].dropna()
                    sub_df['time_sec'] = sub_df[time_col].astype(float).apply(np.floor).astype(int)

                    grouped = sub_df.groupby('time_sec')[signal_col].agg(['min', 'max', 'mean', 'std', 'median'])

                    signal_base = signal_col.strip()
                    grouped.columns = [f'{signal_base}_min', f'{signal_base}_max',
                                       f'{signal_base}_mean', f'{signal_base}_std', f'{signal_base}_median']

                    signal_agg_list.append(grouped)

                except Exception as inner_e:
                    print(f"⚠️ Skipped ({time_col}, {signal_col}) due to error: {inner_e}")

            if signal_agg_list:
                aggregated_data = pd.concat(signal_agg_list, axis=1)
                aggregated_data.reset_index(inplace=True)

                output_filename = filename.replace(".csv", "_downsampled_1Hz.csv")
                output_path = os.path.join(output_folder, output_filename)
                aggregated_data.to_csv(output_path, index=False)
                print(f"✅ Saved downsampled file: {output_filename}")
            else:
                print(f"⚠️ No valid signal pairs found in: {filename}")

        except Exception as e:
            print(f"❌ Failed to process {filename}: {e}")
