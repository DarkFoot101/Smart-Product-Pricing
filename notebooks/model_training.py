# ============================================================
# MODEL PIPELINE : PRICE PREDICTION WITH XGB + TEXT + VISION + LLM
# ============================================================

# -------------------------
# 1. IMPORTS
# -------------------------
import os
import sys
import pickle
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)

# Custom modules
from text_parser import (
    extract_value_unit,
    normalize_to_standard_unit,
    extract_pack_size,
    compute_total_quantity
)
from embeddings import TextEmbedder


# -------------------------
# 2. GLOBAL CONFIG
# -------------------------
RANDOM_STATE = 42

NUM_FEATURES = 2
TEXT_EMBED_DIM = 384
VISION_EMBED_DIM = 1280

TARGET_WEIGHTS = {
    "num": 0.35,
    "text": 0.45,
    "vision": 0.20
}

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"


# -------------------------
# 3. UTILITY FUNCTIONS
# -------------------------
def get_filename(url):
    if pd.isna(url) or url == "":
        return None
    return url.split("/")[-1]


def calculate_smape(actual, predicted):
    denominator = (np.abs(actual) + np.abs(predicted)) / 2.0
    diff = np.abs(actual - predicted) / np.maximum(denominator, 1e-9)
    return 100 * np.mean(diff)


def calculate_smape_log(y_true_log, y_pred_log):
    return calculate_smape(np.expm1(y_true_log), np.expm1(y_pred_log))


# -------------------------
# 4. FEATURE ENGINEERING
# -------------------------
def build_numerical_features(df):
    df = df.copy()

    df["pack_size"] = df["catalog_content"].apply(extract_pack_size)

    extracted = df["catalog_content"].apply(extract_value_unit)
    df["raw_value"] = extracted.apply(lambda x: x[0])
    df["unit"] = extracted.apply(lambda x: x[1])

    df["normalized_value"] = df.apply(
        lambda r: normalize_to_standard_unit(r["raw_value"], r["unit"]), axis=1
    )

    df["total_quantity"] = df.apply(
        lambda r: compute_total_quantity(r["normalized_value"], r["pack_size"]), axis=1
    )

    df["total_quantity"].fillna(df["total_quantity"].median(), inplace=True)
    df["pack_size"].fillna(1, inplace=True)

    df["log_total_quantity"] = np.log1p(df["total_quantity"])

    return df


# -------------------------
# 5. EMBEDDING PIPELINES
# -------------------------
def build_text_embeddings(df, embedder):
    return embedder.get_embeddings(df["catalog_content"].tolist())


def build_vision_embeddings(df, vision_dict):
    vision_vectors = []

    for _, row in df.iterrows():
        fname = get_filename(row.get("image_link", ""))
        if fname in vision_dict and np.array(vision_dict[fname]).shape == (VISION_EMBED_DIM,):
            vision_vectors.append(vision_dict[fname])
        else:
            vision_vectors.append(np.zeros(VISION_EMBED_DIM))

    return np.vstack(vision_vectors)


# -------------------------
# 6. FEATURE SCALING & WEIGHTING
# -------------------------
def apply_weighted_scaling(X, scaler=None, fit=False):
    if fit:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    total_features = NUM_FEATURES + TEXT_EMBED_DIM + VISION_EMBED_DIM

    w_num = np.sqrt((TARGET_WEIGHTS["num"] * total_features) / NUM_FEATURES)
    w_text = np.sqrt((TARGET_WEIGHTS["text"] * total_features) / TEXT_EMBED_DIM)
    w_vision = np.sqrt((TARGET_WEIGHTS["vision"] * total_features) / VISION_EMBED_DIM)

    X[:, :NUM_FEATURES] *= w_num
    X[:, NUM_FEATURES:NUM_FEATURES + TEXT_EMBED_DIM] *= w_text
    X[:, NUM_FEATURES + TEXT_EMBED_DIM:] *= w_vision

    return X, scaler, (w_num, w_text, w_vision)


# -------------------------
# 7. MODEL TRAINING
# -------------------------
def train_xgb(X, y):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.1, random_state=RANDOM_STATE
    )

    model = XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.4,
        reg_alpha=15,
        reg_lambda=2,
        tree_method="hist",
        device="cuda",
        random_state=RANDOM_STATE,
        objective="reg:squarederror"
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=100
    )

    return model


# -------------------------
# 8. LLM PRICE CALIBRATION
# -------------------------
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16
    )

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=15,
        do_sample=False,
        return_full_text=False
    )


def aggressive_calibrate(llm, text, price):
    prompt = (
        f"<|user|>Classify quantity: \"{text[:200]}\"\n"
        "Tags: TAG_SINGLE, TAG_PACK, TAG_BULK. Output ONLY tag.<|end|><|assistant|>"
    )

    try:
        tag = llm(prompt)[0]["generated_text"]

        if "TAG_BULK" in tag:
            return price * 6.7
        if "TAG_PACK" in tag:
            return price * 3.9
        return price * 1.7

    except Exception:
        return price


# -------------------------
# 9. END-TO-END TRAINING SCRIPT
# -------------------------
def main():
    # Load data
    train_df = pd.read_csv("train.csv")

    # Feature engineering
    train_df = build_numerical_features(train_df)

    # Embeddings
    embedder = TextEmbedder()
    X_text = build_text_embeddings(train_df, embedder)

    with open("train_image_embeddings.pkl", "rb") as f:
        vision_dict = pickle.load(f)

    X_vision = build_vision_embeddings(train_df, vision_dict)
    X_num = train_df[["pack_size", "log_total_quantity"]].values

    X = np.hstack([X_num, X_text, X_vision])
    y = np.log1p(train_df["price"])

    # Scaling
    X_scaled, scaler, weights = apply_weighted_scaling(X, fit=True)

    # Train
    model = train_xgb(X_scaled, y)

    # Save artifacts
    joblib.dump(model, "xgb_price_model.pkl")
    joblib.dump(scaler, "feature_scaler.pkl")
    joblib.dump(weights, "weight_factors.pkl")

    print("Pipeline saved successfully.")


if __name__ == "__main__":
    main()
