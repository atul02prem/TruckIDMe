import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from D_tree_new_use import train_decision_tree
from Random_forest_new import train_random_forest
from Knn_new import train_knn
from MLP_new import train_mlp_keras

from plotter import plot_multimodel_multiclass_roc   

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

def eval_and_save(model, X_test, y_test, label_encoder, outdir="figures", prefix="mlp_keras"):
    os.makedirs(outdir, exist_ok=True)

    if hasattr(model, "_train_scaler"):
        Xs = model._train_scaler.transform(X_test)
        probs = model.predict(Xs, verbose=0)
        y_pred = probs.argmax(axis=1)
    else:  
        probs = model.predict_proba(X_test)
        y_pred = probs.argmax(axis=1)

    report_txt = classification_report(
        y_test, y_pred, target_names=label_encoder.classes_, digits=4, zero_division=0
    )
    with open(os.path.join(outdir, f"{prefix}_classification_report.txt"), "w") as f:
        f.write(report_txt)
    print(f"Saved classification report → {outdir}/{prefix}_classification_report.txt")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, xticks_rotation=90, cmap="Blues", colorbar=False)
    plt.title(f"{prefix} — Confusion Matrix")
    plt.savefig(os.path.join(outdir, f"{prefix}_confusion_matrix.png"), dpi=300, bbox_inches="tight")
    plt.close()

    return y_pred, probs

train_df = pd.read_csv("train_70_6+t.csv")
test_df  = pd.read_csv("test_30_6+t.csv")

class ProbaWrapper:
    def __init__(self, model, T=1.0):
        self.m = model   
        self.T = T      

    def predict_proba(self, X):
        if hasattr(self.m, "predict_proba"):
            return self.m.predict_proba(X)
        preds = self.m.predict(X, verbose=0) if hasattr(self.m, "predict") else self.m(X, training=False).numpy()
        import tensorflow as tf
        return tf.nn.softmax(preds / self.T).numpy()
    

WINDOW_LEN = 120
META_COLS = {"target", "window_id"}
FEATURE_COLS = [c for c in train_df.columns if c not in META_COLS]

def prepare_dataset(df, feature_cols, window_len=WINDOW_LEN):
    X_list, y_list = [], []
    for (driver, win_id), g in df.groupby(["target", "window_id"], sort=False):
        g = g.sort_values("time", kind="mergesort")
        if len(g) != window_len:
            continue
        X_list.append(g[feature_cols].to_numpy(dtype=np.float32).reshape(-1))
        y_list.append(driver)
    return np.vstack(X_list).astype(np.float32), np.array(y_list)

X_train, y_train_raw = prepare_dataset(train_df, FEATURE_COLS)
X_test,  y_test_raw  = prepare_dataset(test_df,  FEATURE_COLS)

le = LabelEncoder()
y_train = le.fit_transform(y_train_raw)
y_test  = le.transform(y_test_raw)

n_classes = len(le.classes_)


dt  = train_decision_tree(X_train, y_train)
rf  = train_random_forest(X_train, y_train)
knn = train_knn(X_train, y_train)
mlp = train_mlp_keras(X_train, y_train, n_classes)  
mlp_for_roc = ProbaWrapper(mlp)

eval_and_save(mlp, X_test, y_test, le, prefix="mlp_keras")

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
probs = mlp_for_roc.predict_proba(X_test)        
y_bin = label_binarize(y_test, classes=np.arange(n_classes))
print("MLP micro AUC:", roc_auc_score(y_bin, probs, average="micro", multi_class="ovr"))
print("MLP macro AUC:", roc_auc_score(y_bin, probs, average="macro", multi_class="ovr"))

for name, m in [("DT", dt), ("RF", rf), ("KNN", knn)]:
    if hasattr(m, "classes_"):
        assert np.array_equal(m.classes_, np.arange(n_classes)), f"{name}: classes_ out of order!"

models = {
    "DecisionTree": dt,
    "RandomForest": rf,
    "KNN": knn,
    "MLP": mlp_for_roc    
}

roc_results = plot_multimodel_multiclass_roc(
    models, X_test, y_test, le,
    show_macro=True, show_micro=True,
    title="ROC Comparison: DT vs RF vs KNN vs MLP",
    savepath="figures/all_models_roc.png"
)

for model_name, stats in roc_results.items():
    print(f"{model_name}: micro AUC={stats['micro_auc']:.3f}, macro AUC={stats['macro_auc']:.3f}")