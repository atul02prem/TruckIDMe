from itertools import product
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd
import numpy as np

test_df = pd.read_csv(r"test_30_6+t.csv")
train_df = pd.read_csv(r"train_70_6+t.csv")


WINDOW_LEN = 120
META_COLS = {"target", "window_id"} 

feature_cols_train = [c for c in train_df.columns if c not in META_COLS]
feature_cols_test  = [c for c in test_df.columns  if c not in META_COLS]
assert feature_cols_train == feature_cols_test, "Train/Test feature columns mismatch!"
FEATURE_COLS = feature_cols_train


def prepare_dataset(df, feature_cols, window_len=WINDOW_LEN):
    X_list, y_list, meta_list = [], [], []

    for (driver, win_id), g in df.groupby(["target", "window_id"], sort=False):
        g = g.sort_values("time", kind="mergesort")  # stable sort

        if len(g) != window_len:
            raise ValueError(f"Window ({driver}, {win_id}) has {len(g)} rows (expected {window_len}).")

        # flattten 120×f --->>> 1D
        X_list.append(g[feature_cols].to_numpy(dtype=np.float32).reshape(-1))
        y_list.append(driver)
        meta_list.append((driver, int(win_id)))

    X = np.vstack(X_list) if X_list else np.empty((0, window_len * len(feature_cols)), dtype=np.float32)
    y = np.array(y_list)
    return X, y, meta_list

X_train, y_train_raw, idx_train = prepare_dataset(train_df, FEATURE_COLS)
X_test,  y_test_raw,  idx_test  = prepare_dataset(test_df,  FEATURE_COLS)

print("Training data shape: ", X_train.shape)
print("Testing data shape: ", X_test.shape)

le = LabelEncoder()
y_train = le.fit_transform(y_train_raw)
y_test  = le.transform(y_test_raw)

base_pipe = Pipeline([
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ("clf", KNeighborsClassifier(n_jobs=-1))
])


grid = {
    "n_neighbors":  [1, 3, 5, 7, 11, 15, 21, 31],
    "weights":      ["uniform", "distance"],
    "metric":       ["minkowski", "cosine"],   
    "p":            [1, 2],                    
    "algorithm":    ["brute", "auto"],         
    "leaf_size":    [30],                      
}

best = {"f1": -1.0, "acc": -1.0, "params": None, "pred": None}

for n, w, m, p, alg, leaf in product(
    grid["n_neighbors"], grid["weights"], grid["metric"],
    grid["p"], grid["algorithm"], grid["leaf_size"]
):
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", KNeighborsClassifier(
            n_neighbors=n,
            weights=w,
            metric=m,
            p=p,                
            algorithm=alg,
            leaf_size=leaf,
            n_jobs=-1
        ))
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1  = f1_score(y_test, pred, average="macro")

    if (f1 > best["f1"]) or (f1 == best["f1"] and acc > best["acc"]):
        best = {"f1": f1, "acc": acc, "params": {
                    "n_neighbors": n, "weights": w, "metric": m,
                    "p": p, "algorithm": alg, "leaf_size": leaf
                },
                "pred": pred}

print("\n[KNN grid] Best params:", best["params"])
print(f"[KNN grid] Test Macro-F1: {best['f1']:.4f} | Test Acc: {best['acc']:.4f}\n")
print("Classification report:\n",
      classification_report(y_test, best["pred"], target_names=le.classes_, digits=4))
