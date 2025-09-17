import pandas as pd
import numpy as np
from collections import Counter


df = pd.read_csv(r"../data/combined_CAN_dataset_with_time.csv")
df.columns = df.columns.str.strip()

df = df[df['target']!= 'G1_Subject4']

rows_per_driver = df['target'].value_counts().sort_index()
print(rows_per_driver)

df['time'] = df.groupby('target')['time'].transform(lambda x: (x-x.min())/(x.max()-x.min()))


WINDOW_SIZE = 120
OVERLAP     = 60
STRIDE      = WINDOW_SIZE - OVERLAP  
TEST_FRAC   = 0.30                   
GAP_WINDOWS = 1                      

def make_windows_per_driver(gdf, window_size=WINDOW_SIZE, stride=STRIDE):
    gdf = gdf.sort_values('time').reset_index(drop=True)
    windows = []
    win_id = 0
    for start in range(0, len(gdf) - window_size + 1, stride):
        end = start + window_size
        w = gdf.iloc[start:end].copy()
        if len(w) != window_size:
            continue
        w['window_id'] = win_id
        windows.append(w)
        win_id += 1
    return windows

train_segments, test_segments = [], []

for driver_id, g in df.groupby('target', sort=False):
    windows = make_windows_per_driver(g)
    n = len(windows)
    if n == 0:
        continue

    n_test = int(np.floor(TEST_FRAC * n))
    n_test = min(n_test, max(n - GAP_WINDOWS, 0))

    test_end = n_test                     
    train_start = n_test + GAP_WINDOWS    
    train_end = n                         

    # assign
    for i, w in enumerate(windows):
        if 0 <= i < test_end:
            test_segments.append(w)
        elif train_start <= i < train_end:
            train_segments.append(w)

    print(f"{driver_id}: total {n} windows → test={n_test}, gap={GAP_WINDOWS if n_test < n else 0}, train={max(train_end - train_start, 0)}")

test_df  = pd.concat(test_segments,  ignore_index=True) if test_segments  else pd.DataFrame()
train_df = pd.concat(train_segments, ignore_index=True) if train_segments else pd.DataFrame()

if not test_df.empty:
    test_df  = test_df.sort_values(['target', 'window_id', 'time']).reset_index(drop=True)
if not train_df.empty:
    train_df = train_df.sort_values(['target', 'window_id', 'time']).reset_index(drop=True)

train_df.to_csv("train_70_diff.csv", index=False)
test_df.to_csv("test_30_diff.csv", index=False)

from itertools import product

def pairset(d):
    return set(zip(d['target'], d['window_id'])) if not d.empty else set()

train_pairs = pairset(train_df)
test_pairs  = pairset(test_df)

print("\nFinal shapes by rows:")
print("train_df:", train_df.shape)
print("test_df :", test_df.shape)

print("\nLeakage checks (should be 0):")
print("Train ∩ Test:", len(train_pairs & test_pairs))

print("\nWindows per driver (train):", Counter(train_df['target']))
print("\nWindows per driver (test):", Counter(test_df['target']))
