"""
Showcase and Demo script for the Premier League Transfer Value Predictor.
Runs predictions across multiple star players and demonstrates custom player valuation.
"""
import sys
# Ensure Windows terminal handles UTF-8 safely
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
from src.data_loader import load_player_data
from src.model import TransferValueModel
from predict import display_valuation_report


def run_demo():
    print("=" * 75)
    print("       PREMIER LEAGUE TRANSFER VALUE PREDICTOR - DEMO SHOWCASE")
    print("=" * 75)

    df = load_player_data()
    model = TransferValueModel()
    try:
        model.load()
    except FileNotFoundError:
        print("Training model...")
        model.train(df)
        model.save()

    print("\n>>> DEMO 1: Evaluating Top Premier League Stars Across Positions\n")
    sample_stars = [
        "Erling Haaland",
        "Cole Palmer",
        "Bukayo Saka",
        "Declan Rice",
        "Martin Odegaard",
        "William Saliba",
        "Trent Alexander-Arnold",
        "David Raya",
    ]

    summary_rows = []
    for star in sample_stars:
        matches = df[df["player_name"].str.contains(star, case=False, na=False)]
        if not matches.empty:
            row = matches.iloc[0]
            stats = {
                "goals": int(row["goals"]),
                "assists": int(row["assists"]),
                "minutes_played": int(row["minutes_played"]),
                "age": int(row["age"]),
                "position": row["position"],
            }
            pred = model.predict_value(stats)
            actual = float(row["actual_value_eur_m"])
            diff = pred - actual
            summary_rows.append({
                "Player": row["player_name"],
                "Club": row["team"],
                "Position": row["position"],
                "Age": row["age"],
                "Goals": row["goals"],
                "Assists": row["assists"],
                "Minutes": row["minutes_played"],
                "Actual (€M)": f"€{actual:.1f}M",
                "Predicted (€M)": f"€{pred:.1f}M",
                "Diff (€M)": f"{diff:+.1f}M",
            })

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 75)
    print(">>> DEMO 2: Detailed Decomposition for Cole Palmer (Breakout Season)")
    print("=" * 75)
    palmer_stats = {"goals": 22, "assists": 11, "minutes_played": 2617, "age": 22, "position": "Forward"}
    explanation = model.explain_prediction(palmer_stats)
    display_valuation_report(
        player_name="Cole Palmer (2023-24 Season)",
        stats=palmer_stats,
        predicted_val=explanation["final_predicted_value_eur_m"],
        actual_val=80.0,
        explanation=explanation,
    )

    print("\n" + "=" * 75)
    print(">>> DEMO 3: Hypothetical Scouting Profile - The 'Wonderkid' Striker")
    print("=" * 75)
    wonderkid_stats = {"goals": 16, "assists": 8, "minutes_played": 2400, "age": 19, "position": "Forward"}
    wk_explanation = model.explain_prediction(wonderkid_stats)
    display_valuation_report(
        player_name="Hypothetical 19-Year-Old Wonderkid",
        stats=wonderkid_stats,
        predicted_val=wk_explanation["final_predicted_value_eur_m"],
        actual_val=None,
        explanation=wk_explanation,
    )


if __name__ == "__main__":
    run_demo()
