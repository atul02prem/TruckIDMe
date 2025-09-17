import pandas as pd

# === Load dataset ===
df = pd.read_csv("combined_CAN_dataset_cleaned.csv")  # Update path if needed

# === Add time column grouped by driver ===
df['time'] = df.groupby('target').cumcount()

# === Save updated file ===
df.to_csv("combined_CAN_dataset_with_time.csv", index=False)
print(" Added 'time' column and saved as 'combined_CAN_dataset_with_time.csv'")