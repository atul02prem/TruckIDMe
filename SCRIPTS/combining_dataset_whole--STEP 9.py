import pandas as pd
import os

folder_path = "C:\TruckIDMe Project\Imputed_Data_WithTarget"
all_data = []

for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        path = os.path.join(folder_path, filename)
        df = pd.read_csv(path)
        all_data.append(df)

combined_df = pd.concat(all_data, ignore_index=True)
print(f" Combined dataset shape: {combined_df.shape}")



numeric_cols = combined_df.select_dtypes(include='number').columns
combined_df[numeric_cols] = combined_df[numeric_cols].fillna(method='ffill').fillna(method='bfill')

print(combined_df.isnull().sum().sum() == 0 )

combined_df.columns = combined_df.columns.str.strip()


combined_df.to_csv("combined_CAN_dataset_cleaned.csv", index=False)
print(" Combined dataset saved to 'combined_CAN_dataset_cleaned.csv'")

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# # Split into X and y
# X = combined_df.drop(columns=['target'])
# y = combined_df['target']

# # Scale
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# # Train model
# clf = RandomForestClassifier(n_estimators=1000, random_state=42)
# clf.fit(X_train, y_train)

# importances = clf.feature_importances_
# feature_names = X.columns
# importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
# importance_df = importance_df.sort_values(by='Importance', ascending=False)

# print("🔝 Top 20 Important Features:")
# print(importance_df.head(20))

# import pandas as pd

# # Assuming clf is your trained RandomForestClassifier

# feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

# # Get top 40 features
# top_40 = feat_imp.head(40)
# top_40.to_csv("top_40_features.csv", header=["importance"])
# print("✅ Top 40 feature names saved to top_40_features.csv")