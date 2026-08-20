from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import StratifiedKFold


ROOT = Path(r"C:\Users\Papa Offei\Documents\lalang")
REPORT_DIR = ROOT / "reports" / "extended_validation_source_selector"
VAL = ROOT / "Val.csv"
BASELINE_EXP2 = 0.5892166283468145
RANDOM_STATE = 139

CHOICE_FILES = [
    "extended_subset_hybrid_choices.csv",
    "extended_specialist_subset_hybrid_choices.csv",
    "all_model_subset_hybrid_choices.csv",
    "expanded_all_model_subset_hybrid_choices.csv",
    "specialist_hgb_choices.csv",
    "specialist_extra_trees_choices.csv",
    "specialist_random_forest_choices.csv",
    "specialist_ensemble_choices.csv",
    "specialist_weighted_ensemble_choices.csv",
    "expanded_hgb_choices.csv",
    "expanded_extra_trees_choices.csv",
    "expanded_random_forest_choices.csv",
    "expanded_ensemble_choices.csv",
    "expanded_weighted_ensemble_choices.csv",
]

CONFIGS = [
    ("input_len", 3, 40),
    ("input_len", 3, 60),
    ("input_len", 3, 80),
    ("input_len", 3, 120),
    ("input_len", 4, 40),
    ("input_len", 4, 60),
    ("input_len", 4, 80),
    ("input_len", 4, 120),
    ("input_len", 4, 220),
    ("input_len", 5, 40),
    ("input_len", 5, 60),
    ("input_len", 5, 80),
    ("input_len", 5, 120),
    ("input_len", 5, 220),
    ("input_len", 6, 40),
    ("input_len", 6, 60),
    ("input_len", 6, 80),
    ("input_len", 6, 120),
    ("input_char_len", 4, 40),
    ("input_char_len", 4, 60),
    ("input_char_len", 4, 80),
    ("input_char_len", 4, 120),
    ("input_char_len", 4, 160),
    ("input_char_len", 4, 220),
    ("input_char_len", 5, 40),
    ("input_char_len", 5, 60),
    ("input_char_len", 5, 80),
    ("input_char_len", 5, 120),
    ("input_char_len", 5, 160),
    ("input_char_len", 5, 220),
    ("pred_len", 4, 40),
    ("pred_len", 4, 60),
    ("pred_len", 4, 80),
    ("pred_len", 4, 120),
    ("pred_len", 4, 160),
    ("pred_len", 4, 220),
    ("pred_len", 5, 40),
    ("pred_len", 5, 60),
    ("pred_len", 5, 80),
    ("pred_len", 5, 120),
    ("pred_len", 5, 160),
    ("pred_len", 5, 220),
    ("pred_len", 6, 40),
    ("pred_len", 6, 60),
    ("pred_len", 6, 80),
    ("pred_len", 6, 120),
    ("pred_len", 6, 160),
    ("pred_len", 6, 220),
]


def load_choices():
    val = pd.read_csv(VAL)[["ID", "subset", "input"]]
    val["input_len"] = val["input"].astype(str).str.split().map(len)
    val["input_char_len"] = val["input"].astype(str).str.len()
    frames = []
    for name in CHOICE_FILES:
        path = REPORT_DIR / name
        if not path.exists():
            continue
        df = pd.read_csv(
            path,
            usecols=lambda c: c in {"ID", "subset", "source", "prediction", "target_r1"},
        )
        df["selector"] = path.stem.replace("_choices", "")
        frames.append(df)
    all_choices = pd.concat(frames, ignore_index=True)
    return val, all_choices.merge(val[["ID", "input_len", "input_char_len"]], on="ID", how="left")


def add_bins(df, feature, bins):
    tmp = df.copy()
    if feature == "pred_len":
        tmp["pred_len"] = tmp["prediction"].fillna("").astype(str).str.split().map(len)
    tmp["_bin"] = tmp.groupby("subset")[feature].transform(
        lambda s: pd.qcut(s.rank(method="first"), bins, labels=False, duplicates="drop")
    )
    return tmp


def learn_group_picks(train, min_n, fallback="expanded_all_model_subset_hybrid"):
    picks = []
    for (subset, bin_id), group in train.groupby(["subset", "_bin"]):
        stats = (
            group.groupby("selector")
            .agg(score=("target_r1", "mean"), n=("target_r1", "size"))
            .reset_index()
        )
        stats = stats[stats["n"].ge(min_n)]
        selector = fallback if stats.empty else stats.loc[stats["score"].idxmax(), "selector"]
        picks.append({"subset": subset, "_bin": bin_id, "selector": selector})
    return pd.DataFrame(picks)


def apply_picks(scored, ids, picks, base):
    rows = scored[scored["ID"].isin(ids)].merge(picks, on=["subset", "_bin", "selector"], how="inner")
    rows = rows.drop_duplicates("ID")
    if rows["ID"].nunique() != len(ids):
        missing = set(ids) - set(rows["ID"])
        fallback = base[base["ID"].isin(missing)].assign(selector="expanded_all_model_subset_hybrid")
        rows = pd.concat([rows, fallback], ignore_index=True).drop_duplicates("ID")
    return rows


def fixed_config_oof(all_choices, val, config):
    feature, bins, min_n = config
    scored = add_bins(all_choices, feature, bins)
    base = pd.read_csv(REPORT_DIR / "expanded_all_model_subset_hybrid_choices.csv")
    base = add_bins(base.merge(val[["ID", "input_len", "input_char_len"]], on="ID", how="left"), feature, bins)
    ids = val[["ID", "subset"]].reset_index(drop=True)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    chosen = []
    fold_scores = []
    for fold, (tr_i, va_i) in enumerate(folds.split(ids["ID"], ids["subset"]), start=1):
        train_ids = set(ids.iloc[tr_i]["ID"])
        valid_ids = set(ids.iloc[va_i]["ID"])
        picks = learn_group_picks(scored[scored["ID"].isin(train_ids)], min_n)
        fold_rows = apply_picks(scored, valid_ids, picks, base)
        fold_scores.append({"fold": fold, "score": float(fold_rows["target_r1"].mean())})
        chosen.append(fold_rows)
    out = pd.concat(chosen, ignore_index=True).drop_duplicates("ID")
    return out, fold_scores


def nested_config_oof(all_choices, val):
    base = pd.read_csv(REPORT_DIR / "expanded_all_model_subset_hybrid_choices.csv")
    ids = val[["ID", "subset"]].reset_index(drop=True)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    chosen = []
    fold_summaries = []
    scored_cache = {}
    base_cache = {}
    for fold, (tr_i, va_i) in enumerate(folds.split(ids["ID"], ids["subset"]), start=1):
        train_ids = set(ids.iloc[tr_i]["ID"])
        valid_ids = set(ids.iloc[va_i]["ID"])
        best = None
        for config in CONFIGS:
            feature, bins, min_n = config
            key = (feature, bins)
            if key not in scored_cache:
                scored_cache[key] = add_bins(all_choices, feature, bins)
                base_cache[key] = add_bins(
                    base.merge(val[["ID", "input_len", "input_char_len"]], on="ID", how="left"),
                    feature,
                    bins,
                )
            scored = scored_cache[key]
            picks = learn_group_picks(scored[scored["ID"].isin(train_ids)], min_n)
            train_rows = apply_picks(scored, train_ids, picks, base_cache[key])
            score = float(train_rows["target_r1"].mean())
            if best is None or score > best["train_score"]:
                best = {
                    "feature": feature,
                    "bins": bins,
                    "min_n": min_n,
                    "train_score": score,
                    "picks": picks,
                    "scored": scored,
                    "base": base_cache[key],
                }
        fold_rows = apply_picks(best["scored"], valid_ids, best["picks"], best["base"])
        fold_summaries.append(
            {
                "fold": fold,
                "feature": best["feature"],
                "bins": best["bins"],
                "min_n": best["min_n"],
                "train_score": best["train_score"],
                "valid_score": float(fold_rows["target_r1"].mean()),
            }
        )
        chosen.append(fold_rows)
    out = pd.concat(chosen, ignore_index=True).drop_duplicates("ID")
    return out, fold_summaries


def run():
    val, all_choices = load_choices()
    winning_config = ("input_char_len", 5, 80)
    fixed_oof, fixed_folds = fixed_config_oof(all_choices, val, winning_config)
    nested_oof, nested_folds = nested_config_oof(all_choices, val)

    summary = {
        "feature_leakage_audit": {
            "winning_runtime_features": ["subset", "input_char_len"],
            "test_available": True,
            "reference_or_output_required_at_test": False,
            "prediction_text_required_for_winning_feature": False,
        },
        "target_leakage_audit": {
            "full_validation_group_pick_score": 0.630320475330297,
            "uses_full_validation_target_r1_to_pick_group_selectors": True,
            "fixed_config_oof_group_pick_score": float(fixed_oof["target_r1"].mean()),
            "nested_config_oof_score": float(nested_oof["target_r1"].mean()),
            "fixed_config_fold_scores": fixed_folds,
            "nested_config_fold_scores": nested_folds,
        },
        "baseline_exp2": BASELINE_EXP2,
        "fixed_config_gain_vs_exp2": float(fixed_oof["target_r1"].mean() - BASELINE_EXP2),
        "nested_config_gain_vs_exp2": float(nested_oof["target_r1"].mean() - BASELINE_EXP2),
        "verdict": (
            "No direct feature leakage in the winning feature set, but the reported 0.6303205 "
            "is validation-tuned because target_r1 from the full validation set chooses selectors "
            "inside each subset/length bin. OOF group-pick scores are the safer estimates."
        ),
    }
    fixed_oof.to_csv(REPORT_DIR / "length_grouped_selector_fixed_config_oof_choices.csv", index=False)
    nested_oof.to_csv(REPORT_DIR / "length_grouped_selector_nested_config_oof_choices.csv", index=False)
    (REPORT_DIR / "length_grouped_selector_leakage_audit.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
