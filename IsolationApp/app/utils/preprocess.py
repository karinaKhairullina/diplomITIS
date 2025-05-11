from anomaly_detection.config import FEATURE_ALIASES


def map_feature_aliases(df):
    df = df.copy()
    columns_lower = {col.lower(): col for col in df.columns}
    renamed = {}

    for canonical_name, aliases in FEATURE_ALIASES.items():
        if canonical_name in df.columns:
            continue

        for alt_name in aliases:
            alt_name_lower = alt_name.lower()
            if alt_name_lower in columns_lower:
                original_col = columns_lower[alt_name_lower]
                renamed[original_col] = canonical_name
                break

    return df.rename(columns=renamed)
