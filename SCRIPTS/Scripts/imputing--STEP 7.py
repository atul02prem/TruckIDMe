import pandas as pd
import os

input_folder = r"C:\TruckIDMe Project\g80_dropped"
output_folder = r"C:\TruckIDMe Project\imputed"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):  
        input_path = os.path.join(input_folder, filename)
        print(f"📥 Processing: {filename}")

        try:
            df = pd.read_csv(input_path)

            #numeric columns
            numeric_cols = df.select_dtypes(include='number').columns

            df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(method='bfill')

            output_filename = filename.replace("dropped_cols.csv", "imputed.csv")
            if output_filename == filename:
                output_filename = filename.replace(".csv", "_imputed.csv")

            output_path = os.path.join(output_folder, output_filename)
            df.to_csv(output_path, index=False)

            # Confirm
            has_nans = df.isnull().sum().sum() > 0
            print(f"NaNs left in {output_filename}?: {has_nans}")

        except Exception as e:
            print(f" Failed to process {filename} due to: {e}")
