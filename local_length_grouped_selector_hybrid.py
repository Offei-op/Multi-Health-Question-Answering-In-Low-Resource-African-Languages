from pathlib import Path
import json

import pandas as pd


ROOT = Path(r"C:\Users\Papa Offei\Documents\lalang")
REPORT_DIR = ROOT / "reports" / "extended_validation_source_selector"
VAL = ROOT / "Val.csv"
BASELINE_EXP2 = 0.5892166283468145

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
    all_choices = all_choices.merge(val[["ID", "input_len", "input_char_len"]], on="ID", how="left")
    return val, all_choices


def choose_by_group(all_choices, base, feature, bins, min_n):
    tmp = all_choices.copy()
    if feature == "pred_len":
        tmp["pred_len"] = tmp["prediction"].fillna("").astype(str).str.split().map(len)
    tmp["bin"] = tmp.groupby("subset")[feature].transform(
        lambda s: pd.qcut(s.rank(method="first"), bins, labels=False, duplicates="drop")
    )

    picks = []
    for (subset, bin_id), group in tmp.groupby(["subset", "bin"]):
        stats = (
            group.groupby("selector")
            .agg(score=("target_r1", "mean"), n=("target_r1", "size"))
            .reset_index()
        )
        stats = stats[stats["n"].ge(min_n)]
        selector = (
            "expanded_all_model_subset_hybrid"
            if stats.empty
            else stats.loc[stats["score"].idxmax(), "selector"]
        )
        picks.append({"subset": subset, "bin": int(bin_id), "selector": selector})

    pick_df = pd.DataFrame(picks)
    chosen = tmp.merge(pick_df, on=["subset", "bin", "selector"], how="inner").drop_duplicates("ID")
    if chosen["ID"].nunique() != base["ID"].nunique():
        missing = set(base["ID"]) - set(chosen["ID"])
        fallback = base[base["ID"].isin(missing)].assign(selector="expanded_all_model_subset_hybrid")
        chosen = pd.concat([chosen, fallback], ignore_index=True).drop_duplicates("ID")
    return chosen, pick_df


def run():
    val, all_choices = load_choices()
    base = pd.read_csv(REPORT_DIR / "expanded_all_model_subset_hybrid_choices.csv")
    base_score = float(base["target_r1"].mean())

    results = []
    for feature, bins in [
        ("input_len", 3),
        ("input_len", 4),
        ("input_len", 5),
        ("input_len", 6),
        ("input_char_len", 4),
        ("input_char_len", 5),
        ("pred_len", 4),
        ("pred_len", 5),
        ("pred_len", 6),
    ]:
        for min_n in [40, 60, 80, 120, 160, 220]:
            chosen, _ = choose_by_group(all_choices, base, feature, bins, min_n)
            score = float(chosen["target_r1"].mean())
            results.append(
                {
                    "feature": feature,
                    "bins": bins,
                    "min_n": min_n,
                    "score": score,
                    "gain_vs_exp2": score - BASELINE_EXP2,
                    "gain_vs_base": score - base_score,
                }
            )

    leaderboard = pd.DataFrame(results).sort_values("score", ascending=False)
    best = leaderboard.iloc[0].to_dict()
    chosen, pick_df = choose_by_group(
        all_choices,
        base,
        best["feature"],
        int(best["bins"]),
        int(best["min_n"]),
    )
    score = float(chosen["target_r1"].mean())

    leaderboard.to_csv(REPORT_DIR / "length_grouped_selector_leaderboard.csv", index=False)
    chosen.to_csv(REPORT_DIR / "length_grouped_selector_hybrid_choices.csv", index=False)
    summary = {
        "score": score,
        "gain_vs_exp2": score - BASELINE_EXP2,
        "over_expanded_all_model_subset_hybrid": score - base_score,
        "best_config": best,
        "group_picks": pick_df.to_dict("records"),
        "notes": [
            "No Modal compute was used.",
            "This is validation-tuned over selector outputs; mirror carefully before using on test.",
        ],
    }
    (REPORT_DIR / "length_grouped_selector_hybrid_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(leaderboard.head(30).to_string(index=False))


if __name__ == "__main__":
    run()
