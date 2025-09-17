import os
import pandas as pd

imputed_folder = r"C:\TruckIDMe Project\imputed"
output_folder = r"C:\TruckIDMe Project\Imputed_Data_WithTarget"  
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(imputed_folder):
    if filename.endswith(".csv"):
        input_path = os.path.join(imputed_folder, filename)
        print(f"📄 Processing: {filename}")

        # Extract driver
        driver_id = filename.split("_CANdata_imputed")[0]

        try:
            df = pd.read_csv(input_path)
            df['target'] = driver_id  # Add new col

            output_path = os.path.join(output_folder, filename)
            df.to_csv(output_path, index=False)
            print(f" Saved with target column: {output_path}")
        except Exception as e:
            print(f" Error processing {filename}: {e}")