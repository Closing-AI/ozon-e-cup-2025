import numpy as np
from sklearn.preprocessing import StandardScaler, normalize


def preprocess_features(
    binary_feats,
    meta_feats,
    text_feats,
    text_embs,
    img_embs=None,
    fit_scalers=True,
    meta_scaler=None,
    text_scaler=None,
):
    """
    Preprocess features by scaling and normalizing.

    Order of arrays in the returned features list:
        0: binary_feats
        1: meta_feats_scaled
        2: text_presence_flag
        3: text_feats_scaled
        4: text_embs_normed
        5: img_presence_flag (if img_embs is not None)
        6: img_embs_normed (if img_embs is not None)

    Returns:
        Tuple of processed features and scalers.
    """
    # Fit or use provided scalers
    if fit_scalers or meta_scaler is None:
        meta_scaler = StandardScaler().fit(meta_feats)
    meta_feats_scaled = meta_scaler.transform(meta_feats)

    if fit_scalers or text_scaler is None:
        text_scaler = StandardScaler().fit(text_feats)
    text_feats_scaled = text_scaler.transform(text_feats)

    # Presence flags
    text_presence_flag = (np.linalg.norm(text_embs, axis=1, keepdims=True) != 0).astype(np.float32)

    # Safe L2 normalization
    def safe_l2_normalize(X, flag):
        X_normed = np.zeros_like(X)
        mask = flag.squeeze() == 1
        if np.any(mask):
            X_normed[mask] = normalize(X[mask], norm="l2", axis=1)
        return X_normed

    text_embs_normed = safe_l2_normalize(text_embs, text_presence_flag)

    features = [
        binary_feats,
        meta_feats_scaled,
        text_presence_flag,
        text_feats_scaled,
        text_embs_normed,
    ]

    if img_embs is not None:
        img_presence_flag = (np.linalg.norm(img_embs, axis=1, keepdims=True) != 0).astype(np.float32)
        img_embs_normed = safe_l2_normalize(img_embs, img_presence_flag)
        features.extend([img_presence_flag, img_embs_normed])

    return features, meta_scaler, text_scaler
