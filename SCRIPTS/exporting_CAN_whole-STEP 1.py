import os
import pandas as pd


input_folder = r"C:\TruckIDMe Project\Driver Dataset"      
output_folder = r"C:\TruckIDMe Project\CAN_Data"            
os.makedirs(output_folder, exist_ok=True)


last_valid_col = "84 Wheel-Based Vehicle Speed"


for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):
        input_path = os.path.join(input_folder, filename)
        print(f"📥 Processing: {filename}")

        try:
            df = pd.read_csv(input_path)


            if last_valid_col in df.columns:
                last_index = df.columns.get_loc(last_valid_col)
                df_can = df.iloc[:, 1:last_index+1]  

                output_filename = filename.replace(".csv", "_CANdata.csv")
                output_path = os.path.join(output_folder, output_filename)

                df_can.to_csv(output_path, index=False)
                print(f"✅ Exported {df_can.shape[1]} CAN columns to: {output_filename}")
            else:
                print(f"⚠️ '{last_valid_col}' not found in {filename}. Skipping.")

        except Exception as e:
            print(f"❌ Failed to process {filename}: {e}")
