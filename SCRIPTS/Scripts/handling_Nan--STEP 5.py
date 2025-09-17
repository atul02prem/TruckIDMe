import pandas as pd
import os


input_folder = "C:\TruckIDMe Project\Downsampled_data"
output_folder = "C:\TruckIDMe Project\g80"

os.makedirs(output_folder,exist_ok=True)

for filename in os.listdir(input_folder):
    input_path = os.path.join(input_folder,filename)
    print(f" Processing : {filename}")

    threshold = .80



    try:
        df = pd.read_csv(input_path)
        missing_ratio = df.isnull().mean()
        cols_to_drop = missing_ratio[missing_ratio >= threshold].index.tolist()

        df_cleaned = df.drop(columns=cols_to_drop)
        print(f" Dropped {len(cols_to_drop)} columns with ≥80% missing values.")


        output_filename = filename.replace("downsampled_1Hz.csv", "g80.csv")
        output_path = os.path.join(output_folder,output_filename)
        df_cleaned.to_csv(output_path,index=False)

        print(f"Saved cleaned {output_filename}")


    except Exception as e:
        print(f" Failed to process {filename} due to {e}")















# # Load your cleaned file
# df = pd.read_csv("G2S1\G2S1_CAN_downsampled_1Hz.csv")


# # # Calculate % of missing values per column
# # missing_percent = df.isnull().mean() * 100  # mean() gives fraction, multiply by 100 for %
# # missing_percent = missing_percent.sort_values(ascending=False)

# # # Print top N (optional)
# # print(" Missing value percentage by column:\n")
# # for col, percent in missing_percent.items():
# #     if percent > 0:
# #         print(f"{col}: {percent:.2f}%")

        


# ## Load data
# #df = pd.read_csv("CAN_downsampled_1Hz_cleaned.csv")

# # Threshold: 80% missing
# threshold = 0.80

# # Find columns to drop
# missing_ratio = df.isnull().mean()
# cols_to_drop = missing_ratio[missing_ratio >= threshold].index.tolist()

# # Drop them
# df_cleaned = df.drop(columns=cols_to_drop)

# # Save cleaned file
# df_cleaned.to_csv("CAN_downsampled_1Hz_g80.csv", index=False)

# # Report
# print(f" Dropped {len(cols_to_drop)} columns with ≥80% missing values.")
# print(f" Final cleaned dataset saved to: CAN_downsampled_1Hz_cleaned_final.csv")
