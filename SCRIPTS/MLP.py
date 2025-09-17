import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report

# --------------------
# Reproducibility
# --------------------
SEED = 903100
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# --------------------
# Data (same prep you use for trees/RF)
# --------------------
TRAIN_CSV = r"train_70_6+t.csv"
TEST_CSV  = r"test_30_6+t.csv"

WINDOW_LEN = 120
META_COLS = {"target", "window_id"}

train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

FEATURE_COLS = [c for c in train_df.columns if c not in META_COLS]
assert FEATURE_COLS == [c for c in test_df.columns if c not in META_COLS], "Feature mismatch"

def prepare_dataset(df, feature_cols, window_len=WINDOW_LEN):
    X_list, y_list = [], []
    for (driver, win_id), g in df.groupby(["target", "window_id"], sort=False):
        g = g.sort_values("time", kind="mergesort")
        if len(g) != window_len:
            continue
        X_list.append(g[feature_cols].to_numpy(dtype=np.float32).reshape(-1))
        y_list.append(driver)
    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list)
    return X, y

X_train, y_train_raw = prepare_dataset(train_df, FEATURE_COLS)
X_test,  y_test_raw  = prepare_dataset(test_df,  FEATURE_COLS)

le = LabelEncoder()
y_train = le.fit_transform(y_train_raw)
y_test  = le.transform(y_test_raw)
n_classes = len(le.classes_)
input_dim = X_train.shape[1]

# --------------------
# Scale features (fit on train, apply to test)
# --------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# --------------------
# MLP model (no validation set)
# --------------------
def make_mlp(input_dim, n_classes, dropout=0.3):
    inputs = tf.keras.Input(shape=(input_dim,), dtype=tf.float32)
    x = tf.keras.layers.Dense(512, activation=None, kernel_initializer="he_normal")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.Dense(256, activation=None, kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.Dense(128, activation=None, kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    return model

model = make_mlp(input_dim, n_classes, dropout=0.45)

# Your TF version complained about label_smoothing earlier, so keep it off.
loss = tf.keras.losses.SparseCategoricalCrossentropy()

opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])

# Optional: EarlyStopping on TRAINING loss (does not touch test data)
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=50, restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="loss", factor=0.7, patience=28, min_lr=1e-5, verbose=1
    ),
]

history = model.fit(
    X_train, y_train,
    epochs=1000,              # cap high; EarlyStopping will stop earlier
    batch_size=128,
    shuffle=True,
    verbose=1,
    callbacks=callbacks      # no validation_* passed -> no val_loss/val_acc
)

# --------------------
# Final evaluation on the SAME test_30.csv as other models
# --------------------
probs = model.predict(X_test, verbose=0)
y_pred = probs.argmax(axis=1)

acc = accuracy_score(y_test, y_pred)
mf1 = f1_score(y_test, y_pred, average="macro")

print(f"\nMLP — Test Accuracy: {acc:.4f} | Macro-F1: {mf1:.4f}")
print("\nPer-class report:\n",
      classification_report(y_test, y_pred, target_names=le.classes_, digits=4))
