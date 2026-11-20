def load_data():
    from pathlib import Path
    import pandas as pd
    DATA_DIR = next(
        (path for path in (Path("data/processed"), Path("../data/processed")) if path.is_dir()),
        None,
    )
    if DATA_DIR is None:
        raise FileNotFoundError("Could not find data/processed from the notebook working directory.")

    meta = pd.read_csv(DATA_DIR / "meta.csv")
    rnflt = pd.read_csv(DATA_DIR / "rnflt_features.csv")
    if "filename" not in meta.columns or "filename" not in rnflt.columns:
        raise ValueError("Both CSV files must contain filename for row alignment.")
    data = meta.merge(rnflt, on="filename", how="inner", validate="one_to_one")
    if len(data) != len(meta) or len(data) != len(rnflt):
        raise ValueError("Metadata and RNFLT feature filenames/counts do not match.")

    rnflt_columns = [column for column in rnflt.columns if column != "filename"]
    if len(rnflt_columns) != 9:
        raise ValueError(f"Expected 9 RNFLT feature columns, found {len(rnflt_columns)}.")
    feature_sets = {
        "RNFLT + age": ["age", *rnflt_columns],
        "RNFLT only (no age)": rnflt_columns,
    }
    for columns in feature_sets.values():
        data[columns] = data[columns].apply(pd.to_numeric, errors="coerce")
    data["label"] = data["glaucoma"].astype(str).str.strip().str.lower().map({"no": 0, "yes": 1})
    data["race"] = data["race"].astype(str).str.strip().str.lower()
    if data["label"].isna().any():
        raise ValueError("glaucoma must contain only 'yes' and 'no'.")
    if "split" in data.columns:
        data["split"] = data["split"].astype(str).str.strip().str.lower().replace({"validation": "val"})
    elif "use" in data.columns:
        data["split"] = data["use"].astype(str).str.strip().str.lower().map(
            {"training": "train", "validation": "val", "test": "test"}
        )
    else:
        raise ValueError("meta.csv must contain split or use.")
    if data["split"].isna().any() or set(data["split"].unique()) != {"train", "val", "test"}:
        raise ValueError("Expected official train, val, and test splits.")

    print(f"Data: {DATA_DIR.resolve()} | rows by split: {data['split'].value_counts().to_dict()}")
    print(f"RNFLT predictors: {rnflt_columns}")
    print(f"Missing feature values: {int(data[[*feature_sets['RNFLT + age']]].isna().sum().sum())}")
    print(data.head())
    return data, feature_sets