import pandas as pd
import numpy as np


train_df = pd.read_csv("train_70_diff.csv")
test_df = pd.read_csv("test_30_diff.csv")

FEATURE_COLS = [ "185 Engine Average Fuel Economy_median", 
                "3721 Aftertreatment 1 Diesel Particulate Filter Time Since Last Active Regeneration_median", 
                "5466 Aftertreatment 1 Diesel Particulate Filter Soot Load Regeneration Threshold_median", 
                "1761 Aftertreatment 1 Diesel Exhaust Fluid Tank Volume_median", 
                "3031 Aftertreatment 1 Diesel Exhaust Fluid Tank Temperature 1_median", 
                "1172 Engine Turbocharger 1 Compressor Intake Temperature_median", "target",'time','window_id']

train_df = train_df[[col for col in train_df.columns if col in FEATURE_COLS]].copy()
test_df = test_df[[col for col in test_df.columns if col in FEATURE_COLS]].copy()

train_df.to_csv("train_70_diff_6+t.csv")
test_df.to_csv("test_30_diff_6+t.csv")


