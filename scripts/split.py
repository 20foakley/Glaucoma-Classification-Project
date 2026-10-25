import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

P = "data/processed/meta.csv"
meta = pd.read_csv(P)

tr_idx = np.where(meta.split == "train")[0]
strat = meta.loc[tr_idx, "glaucoma"].astype(str) + "_" + meta.loc[tr_idx, "race"]
dev_train, dev_val = train_test_split(tr_idx, test_size=0.2, stratify=strat, random_state=42)

meta["dev_split"] = meta["split"]
meta.loc[dev_train, "dev_split"] = "dev_train"
meta.loc[dev_val, "dev_split"] = "dev_val"
meta.to_csv(P, index=False)
print(meta.dev_split.value_counts().to_dict())   # 1680 / 420 / 300 / 900