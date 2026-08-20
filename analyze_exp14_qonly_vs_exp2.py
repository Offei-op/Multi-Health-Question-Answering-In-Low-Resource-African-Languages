from collections import Counter
from pathlib import Path
import json

import pandas as pd


ROOT = Path(r"C:\Users\Papa Offei\Documents\lalang")
EXP2 = ROOT / "modal_outputs" / "exp2_crossencoder_rerank" / "val_predictions.csv"
QONLY = ROOT / "modal_outputs" / "exp14_qonly_gemma_reranker_recovered"
OUT = ROOT / "reports" / "exp14_qonly_vs_exp2_complementarity.csv"


def rouge1(pred, ref):
    pred_toks = str(pred).split()
    ref_toks = str(ref).split()
    if not pred_toks or not ref_toks:
        return 0.0
    overlap = sum((Counter(pred_toks) & Counter(ref_toks)).values())
    if overlap == 0:
        return 0.0
    return float(2.0 * overlap / (len(pred_toks) + len(ref_toks)))


def main():
    exp2 = pd.read_csv(EXP2)
    qonly = pd.read_csv(QONLY)
    df = exp2[["ID", "subset", "rerank", "top1", "oracle", "reference"]].merge(
        qonly[["ID", "qonly_bgem3"]], on="ID", how="inner"
    )

    for col in ["rerank", "top1", "oracle", "qonly_bgem3"]:
        df[f"{col}_r1"] = [rouge1(pred, ref) for pred, ref in zip(df[col], df["reference"])]

    df["best_exp2_qonly_r1"] = df[["rerank_r1", "qonly_bgem3_r1"]].max(axis=1)
    df["winner"] = [
        "qonly" if q > e else "exp2" if e > q else "tie"
        for e, q in zip(df["rerank_r1"], df["qonly_bgem3_r1"])
    ]

    summary = {col: float(df[f"{col}_r1"].mean()) for col in ["top1", "rerank", "qonly_bgem3", "oracle"]}
    summary["oracle_between_exp2_qonly"] = float(df["best_exp2_qonly_r1"].mean())
    summary["gain_if_perfect_choose_qonly_or_exp2"] = summary["oracle_between_exp2_qonly"] - summary["rerank"]
    summary["qonly_wins_rows"] = int((df["winner"] == "qonly").sum())
    summary["exp2_wins_rows"] = int((df["winner"] == "exp2").sum())
    summary["ties"] = int((df["winner"] == "tie").sum())

    per = df.groupby("subset")[["rerank_r1", "qonly_bgem3_r1", "best_exp2_qonly_r1", "oracle_r1"]].mean()
    per["perfect_choose_gain"] = per["best_exp2_qonly_r1"] - per["rerank_r1"]
    per["qonly_win_rate"] = df.assign(qwin=df["winner"].eq("qonly")).groupby("subset")["qwin"].mean()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print(json.dumps(summary, indent=2))
    print("PER SUBSET")
    print(per.round(4).to_string())
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
