"""
Inference CLI: Predict market value for any player (lookup existing star or input custom stats).
Compares the model's prediction to the actual transfer value and explains the breakdown.
"""
import argparse
import sys
# Ensure Windows terminal handles UTF-8 safely
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from typing import Optional
import pandas as pd

import config
from src.data_loader import load_player_data, get_player_stats
from src.model import TransferValueModel


def format_currency(val: float) -> str:
    return f"€{val:.1f}M"


def display_valuation_report(
    player_name: str,
    stats: dict,
    predicted_val: float,
    actual_val: Optional[float] = None,
    explanation: Optional[dict] = None,
):
    print("\n" + "=" * 70)
    print(f"  TRANSFER VALUE ASSESSMENT: {player_name.upper()}")
    print("=" * 70)
    print(f" • Position:       {stats['position']}")
    print(f" • Age:            {stats['age']} years old")
    print(f" • Goals Scored:   {stats['goals']}")
    print(f" • Assists:        {stats['assists']}")
    print(f" • Minutes Played: {stats['minutes_played']} mins ({stats['minutes_played'] // 90} matches equivalent)")
    print("-" * 70)

    print(f"\n MODEL PREDICTED VALUE:  {format_currency(predicted_val)}")

    if actual_val is not None:
        diff = predicted_val - actual_val
        pct_diff = (diff / actual_val) * 100.0 if actual_val > 0 else 0.0

        print(f" ACTUAL MARKET VALUE:     {format_currency(actual_val)}")
        print(f" VALUATION DIFFERENCE:   {diff:+.1f}M ({pct_diff:+.1f}%)")

        print("\n MARKET VERDICT:")
        if abs(pct_diff) <= 15.0:
            print("  [FAIRLY VALUED] The market valuation aligns closely with their on-pitch stats.")
        elif diff < 0:
            print(
                f"  [MARKET PREMIUM / OVERPRICED] The player carries a €{abs(diff):.1f}M market premium "
                f"above their pure statistical output (likely due to brand/hype/contract status)."
            )
        else:
            print(
                f"  [BARGAIN / UNDERVALUED] The model estimates they are worth €{diff:.1f}M MORE "
                f"than their current price tag based on their output (high value potential!)."
            )

    if explanation and "contributions" in explanation:
        print("\n" + "-" * 70)
        print(" STATISTICAL DECOMPOSITION (HOW THE MODEL PRICED THIS PLAYER):")
        print("-" * 70)
        for label, contribution in explanation["contributions"].items():
            print(f"   {label:<48} : {contribution:+.2f}M")
        print(f"   {'Raw Linear Model Output':<48} : €{explanation['raw_sum']:.2f}M")
        print(f"   {'Final Estimated Value (min €0.5M)':<48} : {format_currency(predicted_val)}")
    print("=" * 70 + "\n")


def predict_existing_player(model: TransferValueModel, df: pd.DataFrame, player_query: str):
    matches = get_player_stats(player_query, df)
    if matches is None or matches.empty:
        print(f"\n[!] Player matching '{player_query}' not found in the Premier League database.")
        print("    Try another name or search by partial name (e.g. 'Haaland', 'Saka', 'Palmer').")
        return

    for _, row in matches.iterrows():
        stats = {
            "goals": int(row["goals"]),
            "assists": int(row["assists"]),
            "minutes_played": int(row["minutes_played"]),
            "age": int(row["age"]),
            "position": row["position"],
        }
        explanation = model.explain_prediction(stats)
        predicted = explanation["final_predicted_value_eur_m"]
        actual = float(row[config.TARGET])
        title = f"{row['player_name']} ({row.get('team', 'Premier League')} - Season {row.get('season', 'Latest')})"
        display_valuation_report(title, stats, predicted, actual, explanation)


def predict_custom_stats(
    model: TransferValueModel,
    goals: int,
    assists: int,
    minutes_played: int,
    age: int,
    position: str,
    player_name: str = "Custom Player",
    actual_val: Optional[float] = None,
):
    stats = {
        "goals": goals,
        "assists": assists,
        "minutes_played": minutes_played,
        "age": age,
        "position": position.capitalize().strip(),
    }
    explanation = model.explain_prediction(stats)
    predicted = explanation["final_predicted_value_eur_m"]
    display_valuation_report(player_name, stats, predicted, actual_val, explanation)


def run_interactive_mode(model: TransferValueModel, df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("      PREMIER LEAGUE TRANSFER VALUE PREDICTOR (CLI)")
    print("=" * 60)
    print(" 1. Look up an existing Premier League star (e.g. Haaland, Saka, Rice)")
    print(" 2. Enter custom player stats to predict market value")
    print(" 3. Exit")

    choice = input("\nEnter choice (1-3): ").strip()
    if choice == "1":
        query = input("Enter player name to look up: ").strip()
        predict_existing_player(model, df, query)
    elif choice == "2":
        name = input("Player Name: ").strip() or "Custom Player"
        pos = input(f"Position ({'/'.join(config.VALID_POSITIONS)}): ").strip().capitalize()
        if pos not in config.VALID_POSITIONS:
            print(f"Invalid position. Defaulting to Midfielder.")
            pos = "Midfielder"
        try:
            age = int(input("Age: ").strip())
            goals = int(input("Goals in season: ").strip())
            assists = int(input("Assists in season: ").strip())
            minutes = int(input("Minutes played (e.g. 2500): ").strip())
            actual_str = input("Actual transfer/market value if known (€M, optional): ").strip()
            actual = float(actual_str) if actual_str else None
        except ValueError:
            print("Invalid numeric input. Please enter valid integers.")
            return

        predict_custom_stats(model, goals, assists, minutes, age, pos, name, actual)
    else:
        print("Exiting.")


def main():
    parser = argparse.ArgumentParser(description="Predict Premier League Player Transfer Market Value")
    parser.add_argument("--player", type=str, help="Name of existing Premier League player to look up")
    parser.add_argument("--goals", type=int, help="Season goals for custom player")
    parser.add_argument("--assists", type=int, help="Season assists for custom player")
    parser.add_argument("--minutes", type=int, help="Season minutes played for custom player")
    parser.add_argument("--age", type=int, help="Player age")
    parser.add_argument("--position", type=str, choices=config.VALID_POSITIONS, help="Player position")
    parser.add_argument("--actual", type=float, default=None, help="Actual transfer value in €M (optional)")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive CLI menu")

    args = parser.parse_args()

    # Load model
    model = TransferValueModel()
    try:
        model.load()
    except FileNotFoundError:
        print("Model file not found. Training model first...")
        df = load_player_data()
        model.train(df)
        model.save()

    df = load_player_data()

    if args.player:
        predict_existing_player(model, df, args.player)
    elif args.goals is not None and args.assists is not None and args.minutes is not None and args.age is not None and args.position is not None:
        predict_custom_stats(
            model=model,
            goals=args.goals,
            assists=args.assists,
            minutes_played=args.minutes,
            age=args.age,
            position=args.position,
            actual_val=args.actual,
        )
    elif args.interactive or len(sys.argv) == 1:
        run_interactive_mode(model, df)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
