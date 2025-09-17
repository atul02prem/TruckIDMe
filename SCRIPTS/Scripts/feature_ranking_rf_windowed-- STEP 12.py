import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


TRAIN_CSV = r"train_70_diff.csv"   
TEST_CSV  = r"test_30_diff.csv"


WINDOW_LEN = 120
META_COLS = {"target", "window_id"}   # non-feature columns


train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

print(train_df.shape)
print(test_df.shape)


FEATURE_COLS = [c for c in train_df.columns if c not in META_COLS]
assert FEATURE_COLS == [c for c in test_df.columns if c not in META_COLS], "Train/Test feature mismatch"


def prepare_dataset(df: pd.DataFrame, feature_cols, window_len=WINDOW_LEN):
    X_list, y_list, idx = [], [], []

    for (driver, win_id), g in df.groupby(["target", "window_id"], sort=False):
        g = g.sort_values("time", kind="mergesort")
        if len(g) != window_len:
            continue
        X_list.append(g[feature_cols].to_numpy(dtype=np.float32).reshape(-1))
        y_list.append(driver)
        idx.append((driver, int(win_id)))

    X = np.vstack(X_list).astype(np.float32) if X_list else np.empty((0, window_len * len(feature_cols)), dtype=np.float32)
    y = np.array(y_list)
    return X, y, idx

X_train, y_train_raw, train_idx = prepare_dataset(train_df, FEATURE_COLS)
X_test,  y_test_raw,  test_idx  = prepare_dataset(test_df,  FEATURE_COLS)

print(f"Train windows: {X_train.shape} | Test windows: {X_test.shape}")


le = LabelEncoder()
y_train = le.fit_transform(y_train_raw)
y_test  = le.transform(y_test_raw)


rf = RandomForestClassifier(
    n_estimators=24,
    max_depth=None,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=903100,
)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
mf1 = f1_score(y_test, y_pred, average="macro")
print(f"\nRF (window-level) — Test Accuracy: {acc:.4f} | Macro-F1: {mf1:.4f}")
print("\nClassification report:\n",
      classification_report(y_test, y_pred, target_names=le.classes_, digits=4))


flat_names = [f"{feat}_t{t:03d}" for t in range(WINDOW_LEN) for feat in FEATURE_COLS]

imp_flat = pd.Series(rf.feature_importances_, index=flat_names, name="importance").sort_values(ascending=False)

imp_df = imp_flat.reset_index().rename(columns={"index": "feat_time"})
imp_df["sensor"] = imp_df["feat_time"].str.rsplit("_t", n=1).str[0]
imp_by_sensor = imp_df.groupby("sensor")["importance"].sum().sort_values(ascending=False)
imp_by_sensor.to_csv("importances_6030.csv", header=True)

# importances = rf.feature_importances_
# assert len(importances) == len(flat_names), "Length mismatch between importances and flat feature names."

# importance_df = pd.DataFrame({
#     "feature": flat_names,
#     "importance": importances
# }).sort_values("importance", ascending=False)

# top_40 = importance_df.head(40)
# top_40.to_csv("top40_windowed_6030.csv", index=False)

# tmp = top_40.copy()
# tmp["sensor"] = tmp["feature"].str.rsplit("_t", n=1).str[0]
# tmp["t"] = tmp["feature"].str.extract(r"_t(\d+)$").astype(int)
# tmp = tmp[["sensor", "t", "importance"]]
# tmp.to_csv("top40_6030_windowed_with_columns.csv", index=False)



#print("  rf_window_feature_importances_by_sensor.csv")
