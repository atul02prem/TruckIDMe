from itertools import product
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report,confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

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

n_classes = len(le.classes_)


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

knn_model = Pipeline([
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ("knn", KNeighborsClassifier(
        n_neighbors=1,
        weights="uniform",
        metric="minkowski",
        p=1,
        algorithm="brute",
        leaf_size=30,
        n_jobs=-1
    ))
])

knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1m = f1_score(y_test, y_pred, average="macro")
print(f"[KNN fixed] Test Acc: {acc:.4f} | Macro-F1: {f1m:.4f}")

os.makedirs("figures", exist_ok=True)

report_txt = classification_report(y_test, y_pred, target_names=le.classes_, digits=4, zero_division=0)
with open("figures/knn_fixed_classification_report.txt", "w", encoding="utf-8") as f:
    f.write(report_txt)
print("Saved classification report (txt) → figures/knn_fixed_classification_report.txt")

plt.figure(figsize=(10, 0.35*len(le.classes_)+2))
plt.axis("off")
plt.title("KNN (fixed params) — Classification Report", pad=20)
plt.text(0.01, 0.01, report_txt, fontfamily="monospace", fontsize=9)
plt.tight_layout()
plt.savefig("figures/knn_fixed_classification_report.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved classification report (png) → figures/knn_fixed_classification_report.png")

n_classes = len(le.classes_)
cm = confusion_matrix(y_test, y_pred, labels=np.arange(n_classes))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(ax=ax, xticks_rotation=90, cmap="Blues", colorbar=False)
ax.set_title("KNN (fixed params) — Confusion Matrix")
plt.tight_layout()
plt.savefig("figures/knn_fixed_confusion_matrix.png", dpi=300)
plt.close()
print("Saved confusion matrix → figures/knn_fixed_confusion_matrix.png")

cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=le.classes_)
fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(ax=ax, xticks_rotation=90, cmap="Blues", colorbar=True, values_format=".2f")
ax.set_title("KNN (fixed params) — Confusion Matrix (row-normalized)")
plt.tight_layout()
plt.savefig("figures/knn_fixed_confusion_matrix_norm.png", dpi=300)
plt.close()
print("Saved normalized confusion matrix → figures/knn_fixed_confusion_matrix_norm.png")

def train_knn(X_train, y_train):
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("knn", KNeighborsClassifier(
            n_neighbors=1,
            weights="uniform",
            metric="minkowski",
            p=1,
            algorithm="brute",
            leaf_size=30,
            n_jobs=-1
        ))
    ])
    pipe.fit(X_train, y_train)
    return pipe

knn_model = train_knn(X_train, y_train)
