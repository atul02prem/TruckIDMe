import pandas as pd

df = pd.read_csv("combined_CAN_dataset_cleaned_with_time.csv")
print(f"Loaded dataset with shape: {df.shape}")

df.columns = df.columns.str.strip()

rows_per_driver = df.groupby('target').size().reset_index(name='num_rows')
print("\n Number of rows per driver:")
print(rows_per_driver)

print(f"\n Number of columns (excluding target and time col's): {df.drop(columns=['target','time']).shape[1]}")

print(f"\n👤 Total unique drivers: {df['target'].nunique()}")
print(f" Drivers: {df['target'].unique()}")