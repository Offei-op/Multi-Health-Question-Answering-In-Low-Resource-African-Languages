from collections import Counter
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(r"C:\Users\Papa Offei\Documents\lalang")
VAL = ROOT / "Val.csv"
REPORT_DIR = ROOT / "reports" / "deployable_source_selector"

EXP2 = ROOT / "modal_outputs/exp2_crossencoder_rerank/val_predictions.csv"

SOURCES = [
    ("exp2_top1", EXP2, ["top1"]),
    ("exp2_rerank", EXP2, ["rerank"]),
    ("cluster_fast", ROOT / "reports/cluster_aware_selector_experiment/cluster_selector_fast_oof_choices.csv", ["candidate_answer"]),
    ("cluster_fullcap", ROOT / "reports/cluster_aware_selector_experiment/cluster_selector_fullcap_oof_choices.csv", ["candidate_answer"]),
    ("cluster_noleak", ROOT / "reports/cluster_aware_selector_experiment/cluster_selector_noleak_oof_choices.csv", ["candidate_answer"]),
    ("local_regressor", ROOT / "reports/local_candidate_regressor_submission/val_oof_choices.csv", ["candidate_answer"]),
    ("local_regressor_margin", ROOT / "reports/local_candidate_regressor_submission/val_margin_gate_rows.csv", ["candidate_answer"]),
]


def rouge1(pred, ref):
    pred_toks = str(pred).strip().split()
    ref_toks = str(ref).strip().split()
    if not pred_toks or not ref_toks:
        return 0.0
    overlap = sum((Counter(pred_toks) & Counter(ref_toks)).values())
    if overlap == 0:
        return 0.0
    return float(2.0 * overlap / (len(pred_toks) + len(ref_toks)))


def jaccard(a, b):
    sa = set(str(a).split())
    sb = set(str(b).split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def find_col(df, candidates):
    lower = {c.lower(): c for c in df.columns}
    for col in candidates:
        if col in df.columns:
            return col
        if col.lower() in lower:
            return lower[col.lower()]
    return None


def load_sources(val):
    wide = val[["ID", "subset", "output"]].copy()
    loaded = []
    missing = []
    for name, path, candidates in SOURCES:
        if not path.exists():
            missing.append({"source": name, "reason": "missing", "path": str(path)})
            continue
        df = pd.read_csv(path)
        col = find_col(df, candidates)
        if "ID" not in df.columns or col is None:
            missing.append({"source": name, "reason": "bad_columns", "path": str(path), "columns": list(df.columns)})
            continue
        src = df[["ID", col]].drop_duplicates("ID").rename(columns={col: name})
        wide = wide.merge(src, on="ID", how="left")
        wide[name] = wide[name].fillna("")
        loaded.append(name)
    return wide, loaded, missing


def add_source_features(candidates, pred_cols):
    rows = []
    for _, row in candidates.iterrows():
        source = row["source"]
        pred = str(row["prediction"])
        other_preds = [str(row[c]) for c in pred_cols if c != source and str(row[c]) != ""]
        sims = [jaccard(pred, other) for other in other_preds]
        rows.append(
            {
                "ID": row["ID"],
                "subset": row["subset"],
                "source": source,
                "prediction": pred,
                "target_r1": row["target_r1"],
                "pred_len": len(pred.split()),
                "char_len": len(pred),
                "source_nonempty": float(pred != ""),
                "agreement_max": max(sims) if sims else 0.0,
                "agreement_mean": float(np.mean(sims)) if sims else 0.0,
                "agreement_count_ge_080": int(sum(s >= 0.80 for s in sims)),
                "agreement_count_ge_095": int(sum(s >= 0.95 for s in sims)),
                "unique_sources_same_text": int(sum(pred == str(row[c]) for c in pred_cols)),
            }
        )
    return pd.DataFrame(rows)


def make_candidate_table(wide, loaded):
    for name in loaded:
        wide[f"{name}_r1"] = [rouge1(p, r) for p, r in zip(wide[name], wide["output"])]
    frames = []
    for name in loaded:
        tmp = wide[["ID", "subset", "output"] + loaded].copy()
        tmp["source"] = name
        tmp["prediction"] = tmp[name]
        tmp["target_r1"] = wide[f"{name}_r1"]
        frames.append(tmp)
    return add_source_features(pd.concat(frames, ignore_index=True), loaded)


def add_target_encoding(train_df, valid_df, smooth=20.0):
    global_mean = float(train_df["target_r1"].mean())
    valid_df = valid_df.copy()
    for keys, name in [(["source"], "source"), (["subset"], "subset"), (["source", "subset"], "source_subset")]:
        stats = train_df.groupby(keys)["target_r1"].agg(["mean", "count"]).reset_index()
        stats[f"{name}_te"] = (stats["mean"] * stats["count"] + global_mean * smooth) / (stats["count"] + smooth)
        valid_df = valid_df.merge(stats[keys + [f"{name}_te"]], on=keys, how="left")
        valid_df[f"{name}_te"] = valid_df[f"{name}_te"].fillna(global_mean)
    return valid_df


def make_model(kind, cat_cols, num_cols):
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
        ]
    )
    if kind == "hgb":
        reg = HistGradientBoostingRegressor(
            max_iter=220,
            learning_rate=0.04,
            max_leaf_nodes=15,
            l2_regularization=0.08,
            random_state=42,
        )
    elif kind == "extra_trees":
        reg = ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=6,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError(kind)
    return Pipeline([("pre", pre), ("reg", reg)])


def score_choice(cand, col):
    idx = cand.groupby("ID")[col].idxmax()
    chosen = cand.loc[idx].copy()
    return chosen, float(chosen["target_r1"].mean())


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    val = pd.read_csv(VAL)
    val["ID"] = val["ID"].astype(str)
    val["subset"] = val["subset"].astype(str)
    val["output"] = val["output"].fillna("").astype(str)

    wide, loaded, missing = load_sources(val)
    cand = make_candidate_table(wide, loaded)

    source_scores = []
    for name in loaded:
        scores = [rouge1(p, r) for p, r in zip(wide[name], wide["output"])]
        source_scores.append(
            {
                "source": name,
                "coverage": float((wide[name].astype(str) != "").mean()),
                "rouge1": float(np.mean(scores)),
            }
        )
    source_scores = pd.DataFrame(source_scores).sort_values("rouge1", ascending=False)
    wide["oracle_deployable_r1"] = wide[[f"{n}_r1" for n in loaded]].max(axis=1)

    cat_cols = ["subset", "source"]
    exclude = {"ID", "prediction", "target_r1"}
    num_cols_base = [c for c in cand.columns if c not in exclude and c not in cat_cols]
    groups = cand["ID"].to_numpy()
    y = cand["target_r1"].to_numpy()
    gkf = GroupKFold(n_splits=5)

    pred_cols = []
    for kind in ["hgb", "extra_trees"]:
        oof = np.zeros(len(cand), dtype=np.float32)
        for tr, va in gkf.split(cand, y, groups):
            train_df = cand.iloc[tr].copy()
            train_te = add_target_encoding(train_df, train_df.copy())
            valid_te = add_target_encoding(train_df, cand.iloc[va].copy())
            num_cols = num_cols_base + ["source_te", "subset_te", "source_subset_te"]
            model = make_model(kind, cat_cols, num_cols)
            model.fit(train_te[cat_cols + num_cols], train_te["target_r1"].to_numpy())
            oof[va] = model.predict(valid_te[cat_cols + num_cols])
        col = f"{kind}_pred"
        cand[col] = oof
        pred_cols.append(col)

    cand["ensemble_pred"] = cand[pred_cols].mean(axis=1)
    results = []
    choices = {}
    for col in pred_cols + ["ensemble_pred"]:
        chosen, score = score_choice(cand, col)
        label = col.replace("_pred", "")
        choices[label] = chosen
        results.append(
            {
                "label": label,
                "rouge1": score,
                "choice_counts": chosen["source"].value_counts().to_dict(),
                "per_subset": chosen.groupby("subset")["target_r1"].mean().round(6).to_dict(),
            }
        )

    exp2_score = float(source_scores.loc[source_scores["source"] == "exp2_rerank", "rouge1"].iloc[0])
    best_single_score = float(source_scores["rouge1"].iloc[0])
    deployable_oracle = float(wide["oracle_deployable_r1"].mean())
    leaderboard = pd.DataFrame(
        [
            {
                "label": r["label"],
                "rouge1": r["rouge1"],
                "gain_vs_exp2_rerank": r["rouge1"] - exp2_score,
                "gain_vs_best_single": r["rouge1"] - best_single_score,
                "top_choice": max(r["choice_counts"].items(), key=lambda x: x[1])[0],
            }
            for r in results
        ]
    ).sort_values("rouge1", ascending=False)
    best_label = str(leaderboard.iloc[0]["label"])

    cand.to_csv(REPORT_DIR / "candidate_level_oof_scores.csv", index=False)
    wide.to_csv(REPORT_DIR / "wide_source_scores.csv", index=False)
    source_scores.to_csv(REPORT_DIR / "source_scores.csv", index=False)
    leaderboard.to_csv(REPORT_DIR / "selector_leaderboard.csv", index=False)
    choices[best_label].to_csv(REPORT_DIR / "best_oof_choices.csv", index=False)

    summary = {
        "loaded_sources": loaded,
        "missing_sources": missing,
        "best_single_source": source_scores.iloc[0].to_dict(),
        "exp2_rerank": exp2_score,
        "deployable_oracle": deployable_oracle,
        "deployable_oracle_gain_vs_exp2": deployable_oracle - exp2_score,
        "best_selector": leaderboard.iloc[0].to_dict(),
        "selector_results": results,
    }
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(source_scores.to_string(index=False))
    print(leaderboard.to_string(index=False))


if __name__ == "__main__":
    main()
