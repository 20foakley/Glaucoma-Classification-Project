import glob
import numpy as np
import pandas as pd

files = sorted(glob.glob("data/raw/harvard_gf/**/*.npz", recursive=True))
print(len(files), 'npz files found')

d = np.load(files[0], allow_pickle=True)

for k in d.files:
    print(f"{k:15s} shape: {d[k].shape} dtype: {d[k].dtype}")

for c in glob.glob("data/raw/harvard_gf/**/*.csv", recursive=True):
    df = pd.read_csv(c)
    print("\n", c, df.shape)
    print(df.info())
    print(df.columns.tolist())
    print(df.head(3))
