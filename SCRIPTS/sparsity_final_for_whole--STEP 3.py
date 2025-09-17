import pandas as pd
import os

input_folder = r"C:\TruckIDMe Project\Downsampled_data"  

sparsity_table = pd.DataFrame()

for filename in os.listdir(input_folder):
    if filename.endswith("_downsampled_1Hz.csv"):
        file_path = os.path.join(input_folder, filename)
        try:
            df = pd.read_csv(file_path)

            sparsity = df.isnull().mean() * 100

            driver_name = filename.replace("_CANdata_downsampled_1Hz.csv", "")
            sparsity.name = driver_name

            sparsity_table = pd.concat([sparsity_table, sparsity.to_frame().T])

        except Exception as e:
            print(f"⚠️ Error with file {filename}: {e}")

output_csv = "sparsity_table_20May.csv"
sparsity_table.to_csv(output_csv, float_format="%.2f")
print(f"\n✅ Sparsity matrix for all drivers saved to: {output_csv}")
