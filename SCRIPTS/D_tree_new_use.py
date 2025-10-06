import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report


train_df = pd.read_csv(r"train_70_6+t.csv")
test_df  = pd.read_csv(r"test_30_6+t.csv")

WINDOW_LEN = 120
META_COLS = {"target", "window_id"}

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

print("Train shape:", X_train.shape, y_train.shape)
print("Test  shape:", X_test.shape,  y_test.shape)

clf = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=7,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=0.02,
        class_weight="balanced",
        random_state=903100
    )
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


print("\nClassification report for Decision Tree:")
print(classification_report(y_test, y_pred, target_names=le.classes_, digits=4))

from sklearn.tree import DecisionTreeClassifier

def train_decision_tree(X_train, y_train):
    clf = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=7,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=0.02,
        class_weight="balanced",
        random_state=903100
    )
    clf.fit(X_train, y_train)
    return clf


