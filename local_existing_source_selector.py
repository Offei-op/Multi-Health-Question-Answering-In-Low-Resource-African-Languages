from collections import Counter
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(r"C:\Users\Papa Offei\Documents\lalang")
AUDIT_DIR = ROOT / "reports" / "existing_prediction_oracle_audit"
AUDIT_ROWS = AUDIT_DIR / "existing_prediction_oracle_rows.csv"
REPORT_DIR = ROOT / "reports" / "existing_source_selector"


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


def overlap_f1(a, b):
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    overlap = sum((Counter(ta) & Counter(tb)).values())
    if overlap == 0:
        return 0.0
    return float(2.0 * overlap / (len(ta) + len(tb)))


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


def shape_features(text):
    s = str(text)
    toks = tokens(s)
    chars = len(s)
    return {
        "tok_len": len(toks),
        "char_len": chars,
        "uniq_ratio": len(set(toks)) / max(1, len(toks)),
        "avg_tok_chars": chars / max(1, len(toks)),
        "digit_count": sum(ch.isdigit() for ch in s),
        "punct_count": sum((not ch.isalnum()) and (not ch.isspace()) for ch in s),
        "is_blank": int(len(s.strip()) == 0),
    }


def build_candidate_table():
    wide = pd.read_csv(AUDIT_ROWS)
    pred_cols = [c for c in wide.columns if not c.endswith("_r1") and c not in {"ID", "subset", "output", "oracle_existing_source"}]
    pred_cols = [c for c in pred_cols if f"{c}_r1" in wide.columns]
    reference_sources = [c for c in ["exp2_rerank", "exp3_top100_rerank", "exp5_encoder_exp2_rerank", "exp14_qonly", "exp9_jina"] if c in pred_cols]

    rows = []
    for _, row in wide.iterrows():
        ref_texts = {src: row[src] for src in reference_sources}
        source_counts = Counter(str(row[src]) for src in pred_cols)
        for src in pred_cols:
            pred = row[src]
            feats = {
                "ID": row["ID"],
                "subset": row["subset"],
                "source": src,
                "prediction": pred,
                "target_r1": row[f"{src}_r1"],
                "source_agreement_count": source_counts[str(pred)],
                "source_agreement_frac": source_counts[str(pred)] / len(pred_cols),
            }
            feats.update(shape_features(pred))
            for ref_src, ref_pred in ref_texts.items():
                feats[f"tok_f1_vs_{ref_src}"] = overlap_f1(pred, ref_pred)
                feats[f"char3_jaccard_vs_{ref_src}"] = char_jaccard(pred, ref_pred)
                feats[f"same_as_{ref_src}"] = int(str(pred) == str(ref_pred))
            rows.append(feats)
    return pd.DataFrame(rows), wide, pred_cols


def make_model(kind, cat_cols, num_cols):
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
        ]
    )
    if kind == "hgb":
        model = HistGradientBoostingRegressor(
            max_iter=240,
            learning_rate=0.035,
            max_leaf_nodes=15,
            l2_regularization=0.08,
            random_state=42,
        )
    elif kind == "extra_trees":
        model = ExtraTreesRegressor(
            n_estimators=220,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )
    elif kind == "rf":
        model = RandomForestRegressor(
            n_estimators=180,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )
    elif kind == "ridge":
        pre = ColumnTransformer(
            [
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
                ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols),
            ]
        )
        model = Ridge(alpha=10.0)
    else:
        raise ValueError(kind)
    return Pipeline([("pre", pre), ("model", model)])


def score_choices(cand, pred_score_col, label):
    idx = cand.groupby("ID")[pred_score_col].idxmax()
    chosen = cand.loc[idx].copy()
    return {
        "label": label,
        "rouge1": float(chosen["target_r1"].mean()),
        "choice_counts": chosen["source"].value_counts().to_dict(),
        "per_subset": chosen.groupby("subset")["target_r1"].mean().round(6).to_dict(),
    }, chosen


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cand, wide, pred_cols = build_candidate_table()
    base_scores = {src: float(wide[f"{src}_r1"].mean()) for src in pred_cols}
    best_single_source = max(base_scores.items(), key=lambda x: x[1])
    oracle_existing_sources = float(wide[[f"{src}_r1" for src in pred_cols]].max(axis=1).mean())

    exclude = {"ID", "prediction", "target_r1"}
    cat_cols = ["subset", "source"]
    num_cols = [c for c in cand.columns if c not in exclude and c not in cat_cols]
    X = cand[cat_cols + num_cols]
    y = cand["target_r1"].to_numpy()
    groups = cand["ID"].to_numpy()

    gkf = GroupKFold(n_splits=5)
    results = []
    choice_files = {}
    pred_cols_oof = []
    for kind in ["hgb", "extra_trees", "ridge"]:
        oof = np.zeros(len(cand), dtype=np.float32)
        for tr, va in gkf.split(X, y, groups):
            model = make_model(kind, cat_cols, num_cols)
            model.fit(X.iloc[tr], y[tr])
            oof[va] = model.predict(X.iloc[va])
        col = f"{kind}_pred_r1"
        cand[col] = oof
        pred_cols_oof.append(col)
        res, chosen = score_choices(cand, col, kind)
        results.append(res)
        choice_files[kind] = chosen

    cand["ensemble_avg_pred_r1"] = cand[pred_cols_oof].mean(axis=1)
    res, chosen = score_choices(cand, "ensemble_avg_pred_r1", "ensemble_avg")
    results.append(res)
    choice_files["ensemble_avg"] = chosen

    result_df = pd.DataFrame(
        [
            {
                "label": r["label"],
                "rouge1": r["rouge1"],
                "gain_vs_best_single": r["rouge1"] - best_single_source[1],
                "gain_vs_exp2_rerank": r["rouge1"] - base_scores.get("exp2_rerank", best_single_source[1]),
                "top_choice": max(r["choice_counts"].items(), key=lambda x: x[1])[0] if r["choice_counts"] else "",
            }
            for r in results
        ]
    ).sort_values("rouge1", ascending=False)
    best_label = str(result_df.iloc[0]["label"])
    best_choices = choice_files[best_label]

    cand.to_csv(REPORT_DIR / "candidate_level_oof_scores.csv", index=False)
    result_df.to_csv(REPORT_DIR / "selector_leaderboard.csv", index=False)
    best_choices.to_csv(REPORT_DIR / "best_oof_choices.csv", index=False)

    summary = {
        "experiment": "existing_source_candidate_level_selector",
        "rows": int(wide.shape[0]),
        "candidate_rows": int(cand.shape[0]),
        "sources": pred_cols,
        "best_single_source": {"source": best_single_source[0], "rouge1": best_single_source[1]},
        "exp2_rerank": base_scores.get("exp2_rerank"),
        "oracle_existing_sources": oracle_existing_sources,
        "oracle_gain_vs_best_single": oracle_existing_sources - best_single_source[1],
        "results": results,
        "best": result_df.iloc[0].to_dict(),
        "notes": [
            "OOF split is by ID, so all candidates for a validation row are held out together.",
            "Features use only source name, subset, prediction text shape, and agreement/similarity among candidate predictions.",
            "This does not require Modal; conversion to test requires the same prediction sources to exist for test.",
        ],
    }
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
