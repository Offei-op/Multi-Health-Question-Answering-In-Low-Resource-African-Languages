from collections import Counter
from pathlib import Path
import json
import math
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(r"C:\Users\Papa Offei\Documents\lalang")
EXP2 = ROOT / "modal_outputs" / "exp2_crossencoder_rerank" / "val_predictions.csv"
QONLY = ROOT / "modal_outputs" / "exp14_qonly_gemma_reranker_recovered"
REPORT_DIR = ROOT / "reports" / "exp14_qonly_exp2_selector"
TODO = ROOT / "reports" / "promising_final_submission_todo_2026-06-08.md"


def tokens(text):
    return str(text).strip().split()


def rouge1(pred, ref):
    pred_toks = tokens(pred)
    ref_toks = tokens(ref)
    if not pred_toks or not ref_toks:
        return 0.0
    overlap = sum((Counter(pred_toks) & Counter(ref_toks)).values())
    if overlap == 0:
        return 0.0
    return float(2.0 * overlap / (len(pred_toks) + len(ref_toks)))


def jaccard(a, b):
    sa, sb = set(tokens(a)), set(tokens(b))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def overlap_f1(a, b):
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    overlap = sum((Counter(ta) & Counter(tb)).values())
    if overlap == 0:
        return 0.0
    return 2.0 * overlap / (len(ta) + len(tb))


def char_jaccard(a, b, n=3):
    def grams(x):
        s = re.sub(r"\s+", " ", str(x).strip().lower())
        if len(s) < n:
            return {s} if s else set()
        return {s[i : i + n] for i in range(len(s) - n + 1)}

    ga, gb = grams(a), grams(b)
    if not ga and not gb:
        return 1.0
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def text_shape(prefix, values):
    rows = {}
    for text in values:
        s = str(text)
        toks = tokens(s)
        chars = len(s)
        uniq = len(set(toks))
        rows.setdefault(f"{prefix}_tok_len", []).append(len(toks))
        rows.setdefault(f"{prefix}_char_len", []).append(chars)
        rows.setdefault(f"{prefix}_uniq_ratio", []).append(uniq / max(1, len(toks)))
        rows.setdefault(f"{prefix}_avg_tok_chars", []).append(chars / max(1, len(toks)))
        rows.setdefault(f"{prefix}_digit_count", []).append(sum(ch.isdigit() for ch in s))
        rows.setdefault(f"{prefix}_punct_count", []).append(sum((not ch.isalnum()) and (not ch.isspace()) for ch in s))
    return rows


def ratio(a, b):
    return a / max(1e-6, b)


def build_frame():
    exp2 = pd.read_csv(EXP2)
    qonly = pd.read_csv(QONLY)
    df = exp2[["ID", "subset", "top1", "rerank", "oracle", "reference"]].merge(
        qonly[["ID", "qonly_bgem3"]], on="ID", how="inner"
    )

    for col in ["top1", "rerank", "qonly_bgem3", "oracle"]:
        df[f"{col}_r1"] = [rouge1(pred, ref) for pred, ref in zip(df[col], df["reference"])]
    df["target_qonly_better"] = (df["qonly_bgem3_r1"] > df["rerank_r1"]).astype(int)
    df["target_gain_if_qonly"] = df["qonly_bgem3_r1"] - df["rerank_r1"]

    for prefix, col in [("top1", "top1"), ("exp2", "rerank"), ("qonly", "qonly_bgem3")]:
        for name, vals in text_shape(prefix, df[col]).items():
            df[name] = vals

    pairs = [
        ("exp2", "qonly", "rerank", "qonly_bgem3"),
        ("exp2", "top1", "rerank", "top1"),
        ("qonly", "top1", "qonly_bgem3", "top1"),
    ]
    for a_name, b_name, a_col, b_col in pairs:
        df[f"{a_name}_{b_name}_same_text"] = (df[a_col].astype(str) == df[b_col].astype(str)).astype(int)
        df[f"{a_name}_{b_name}_tok_f1"] = [overlap_f1(a, b) for a, b in zip(df[a_col], df[b_col])]
        df[f"{a_name}_{b_name}_tok_jaccard"] = [jaccard(a, b) for a, b in zip(df[a_col], df[b_col])]
        df[f"{a_name}_{b_name}_char3_jaccard"] = [char_jaccard(a, b) for a, b in zip(df[a_col], df[b_col])]

    for stem in ["tok_len", "char_len", "uniq_ratio", "avg_tok_chars", "digit_count", "punct_count"]:
        df[f"qonly_minus_exp2_{stem}"] = df[f"qonly_{stem}"] - df[f"exp2_{stem}"]
        df[f"qonly_div_exp2_{stem}"] = [ratio(a, b) for a, b in zip(df[f"qonly_{stem}"], df[f"exp2_{stem}"])]
        df[f"exp2_minus_top1_{stem}"] = df[f"exp2_{stem}"] - df[f"top1_{stem}"]
        df[f"qonly_minus_top1_{stem}"] = df[f"qonly_{stem}"] - df[f"top1_{stem}"]

    return df


def make_model(kind, cat_cols, num_cols):
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
        ]
    )
    if kind == "hgb":
        clf = HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.045,
            max_leaf_nodes=15,
            l2_regularization=0.05,
            random_state=42,
        )
    elif kind == "extra_trees":
        clf = ExtraTreesClassifier(
            n_estimators=600,
            min_samples_leaf=12,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
    elif kind == "rf":
        clf = RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=15,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
    elif kind == "logreg":
        pre = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
                ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols),
            ]
        )
        clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5)
    else:
        raise ValueError(kind)
    return Pipeline([("pre", pre), ("clf", clf)])


def evaluate_selector(df, proba, label):
    base = float(df["rerank_r1"].mean())
    best = {
        "label": label,
        "base_exp2_r1": base,
        "qonly_r1": float(df["qonly_bgem3_r1"].mean()),
        "oracle_between_exp2_qonly": float(df[["rerank_r1", "qonly_bgem3_r1"]].max(axis=1).mean()),
        "best_threshold": None,
        "selected_r1": -1.0,
        "gain_vs_exp2": None,
        "qonly_pick_rate": None,
        "accuracy": None,
        "auc": None,
    }
    for threshold in np.linspace(0.05, 0.95, 91):
        choose = proba >= threshold
        selected = np.where(choose, df["qonly_bgem3_r1"], df["rerank_r1"])
        score = float(np.mean(selected))
        if score > best["selected_r1"]:
            best["selected_r1"] = score
            best["best_threshold"] = float(threshold)
            best["gain_vs_exp2"] = score - base
            best["qonly_pick_rate"] = float(np.mean(choose))
            best["accuracy"] = float(accuracy_score(df["target_qonly_better"], choose))
    try:
        best["auc"] = float(roc_auc_score(df["target_qonly_better"], proba))
    except ValueError:
        best["auc"] = None
    return best


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_frame()
    feature_exclude = {
        "ID",
        "top1",
        "rerank",
        "qonly_bgem3",
        "oracle",
        "reference",
        "top1_r1",
        "rerank_r1",
        "qonly_bgem3_r1",
        "oracle_r1",
        "target_qonly_better",
        "target_gain_if_qonly",
    }
    cat_cols = ["subset"]
    num_cols = [c for c in df.columns if c not in feature_exclude and c not in cat_cols]
    X = df[cat_cols + num_cols]
    y = df["target_qonly_better"].to_numpy()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    oof_cols = {}
    for kind in ["hgb", "extra_trees", "rf", "logreg"]:
        proba = np.zeros(len(df), dtype=np.float32)
        for tr, va in cv.split(X, y):
            model = make_model(kind, cat_cols, num_cols)
            model.fit(X.iloc[tr], y[tr])
            proba[va] = model.predict_proba(X.iloc[va])[:, 1]
        oof_cols[f"{kind}_proba_qonly"] = proba
        results.append(evaluate_selector(df, proba, kind))

    avg_proba = np.mean([oof_cols[k] for k in oof_cols], axis=0)
    oof_cols["ensemble_avg_proba_qonly"] = avg_proba
    results.append(evaluate_selector(df, avg_proba, "ensemble_avg"))

    result_df = pd.DataFrame(results).sort_values("selected_r1", ascending=False)
    best = result_df.iloc[0].to_dict()

    out_df = df.copy()
    for col, vals in oof_cols.items():
        out_df[col] = vals
    out_df.to_csv(REPORT_DIR / "oof_predictions.csv", index=False)
    result_df.to_csv(REPORT_DIR / "selector_results.csv", index=False)

    summary = {
        "experiment": "exp14_qonly_vs_exp2_selector",
        "rows": int(len(df)),
        "feature_count": int(len(cat_cols) + len(num_cols)),
        "categorical_features": cat_cols,
        "numeric_feature_count": int(len(num_cols)),
        "results": results,
        "best": best,
        "notes": [
            "OOF selector uses only deployable prediction/text-shape features; no reference-derived features are included.",
            "Target is whether q-only prediction has higher ROUGE-1 than exp2 q+a prediction.",
            "Threshold is selected on OOF predictions, so treat gain as exploratory but less leaky than in-sample fitting.",
        ],
    }
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if best["gain_vs_exp2"] and best["gain_vs_exp2"] > 0:
        entry = (
            "\n## Newly Proven Local Selector Candidate\n\n"
            f"- Exp14 q-only-vs-exp2 selector: best OOF `{best['label']}` selects q-only at threshold "
            f"`{best['best_threshold']:.2f}` with ROUGE-1 `{best['selected_r1']:.6f}` vs exp2 "
            f"`{best['base_exp2_r1']:.6f}` (`+{best['gain_vs_exp2']:.6f}`). "
            "This should be considered for final source-aware selector work, but it needs a train+val/test-time "
            "q-only reranker artifact before conversion to submission.\n"
        )
        with TODO.open("a", encoding="utf-8") as f:
            f.write(entry)


if __name__ == "__main__":
    main()
