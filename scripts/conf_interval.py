import numpy as np
import scipy.stats as stats

from load_data import load_data

def confidence_interval(data, confidence=0.95):
    res = stats.bootstrap(
        (y_true,y_pred),
        np.mean,
        confidence_level=confidence,
        n_resamples=10000,
        method = 'percentile',
    )
    return res.confidence_interval

if __name__ == "__main__":
    data, feature_sets  = load_data()
    print(data)
    test_mask = data["split"].eq("test").to_numpy()
    y = data["label"].to_numpy(dtype=np.int64)
    ci = confidence_interval(data["label"].to_numpy())
    print(ci)