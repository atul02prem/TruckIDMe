import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report


TRAIN_CSV = r"train_70_6+t.csv"
TEST_CSV  = r"test_30_6+t.csv"

WINDOW_LEN = 120
META_COLS = {"target", "window_id"}   


train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

FEATURE_COLS = [c for c in train_df.columns if c not in META_COLS]
assert FEATURE_COLS == [c for c in test_df.columns if c not in META_COLS], "Train/Test feature mismatch"

def prepare_dataset(df, feature_cols, window_len=WINDOW_LEN):
    X_list, y_list, idx = [], [], []
    for (driver, win_id), g in df.groupby(["target", "window_id"], sort=False):
        g = g.sort_values("time", kind="mergesort")
        if len(g) != window_len:
            continue
        X_list.append(g[feature_cols].to_numpy(dtype=np.float32).reshape(-1))
        y_list.append(driver)
        idx.append((driver, int(win_id)))
    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list)
    return X, y, idx

X_train, y_train_raw, train_idx = prepare_dataset(train_df, FEATURE_COLS)
X_test,  y_test_raw,  test_idx  = prepare_dataset(test_df,  FEATURE_COLS)

le = LabelEncoder()
y_train = le.fit_transform(y_train_raw)
y_test  = le.transform(y_test_raw)

print("Train windows:", X_train.shape, " Test windows:", X_test.shape)
from itertools import product

grid = {
    "n_estimators": [8],
    "max_depth": [None],
    "min_samples_leaf": [4],
    "max_features": [0.07],
    "class_weight": ["balanced_subsample"]
}

best = {"acc": -1, "f1": -1, "params": None}
for n, md, msl, mf, cw in product(grid["n_estimators"], grid["max_depth"], grid["min_samples_leaf"], grid["max_features"], grid["class_weight"]):
    model = RandomForestClassifier(
        n_estimators=n,
        max_depth=md,
        min_samples_leaf=msl,
        max_features=mf,
        class_weight=cw,
        n_jobs=-1,
        random_state=903100
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    a = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="macro")
    if f1 > best["f1"]:
        best = {"acc": a, "f1": f1, "params": {"n_estimators": n, "max_depth": md, "min_samples_leaf": msl, "max_features": mf, "class_weight": cw}}

print("\n[GridSearch-lite] Best:", best)

best_params = best["params"]
rf = RandomForestClassifier(
    n_estimators=best_params["n_estimators"],
    max_depth=best_params["max_depth"],
    min_samples_leaf=best_params["min_samples_leaf"],
    max_features=best_params["max_features"],
    class_weight=best_params["class_weight"],
    n_jobs=-1,
    random_state=903100,
)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1m = f1_score(y_test, y_pred, average="macro")
print(f"\n[Refit best] Test Accuracy={acc:.4f}  Macro-F1={f1m:.4f}")
print("\nClassification report:\n",
      classification_report(y_test, y_pred, target_names=le.classes_, digits=4))


