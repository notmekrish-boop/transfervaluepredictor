"""
Script to fetch live 2024-2025/current Premier League players using football-data.org API key.
Saves enriched live player statistics with team crests and model valuations.
"""
import sys
# Ensure Windows terminal handles UTF-8 safely
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import config
from src.api_client import FootballDataClient
from src.model import TransferValueModel


def calculate_age(date_of_birth_str: str) -> int:
    """Calculate player age from YYYY-MM-DD birthdate string."""
    if not date_of_birth_str:
        return 25
    try:
        birth_date = datetime.strptime(date_of_birth_str, "%Y-%m-%d")
        today = datetime.now()
        age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
        return max(16, min(42, age))
    except Exception:
        return 25


def map_position(pos_str: str) -> str:
    """Map raw API position strings to Forward, Midfielder, Defender, Goalkeeper."""
    if not pos_str:
        return "Forward"
    p = pos_str.lower()
    if "goal" in p or "keeper" in p or p == "gk":
        return "Goalkeeper"
    elif "defen" in p or "back" in p or p in ["cb", "lb", "rb"]:
        return "Defender"
    elif "midfield" in p or p in ["cm", "cam", "cdm", "lm", "rm"]:
        return "Midfielder"
    else:
        return "Forward"


def fetch_and_cache_live_data() -> List[Dict[str, Any]]:
    print("Connecting to football-data.org via requests...")
    client = FootballDataClient(api_key=config.FOOTBALL_DATA_API_KEY)

    # 1. Fetch live scorers
    scorers_data = client.get_scorers(competition="PL", limit=50)
    # 2. Fetch teams and squads
    teams_data = client.get_teams(competition="PL")

    teams_map = {}
    player_positions = {}
    if teams_data and "teams" in teams_data:
        for t in teams_data["teams"]:
            t_name = t.get("shortName") or t.get("name")
            t_crest = t.get("crest")
            teams_map[t.get("id")] = {"name": t_name, "crest": t_crest}
            for sq_player in t.get("squad", []):
                p_id = sq_player.get("id")
                player_positions[p_id] = {
                    "position": map_position(sq_player.get("position")),
                    "dateOfBirth": sq_player.get("dateOfBirth"),
                    "team": t_name,
                    "crest": t_crest,
                }

    model = TransferValueModel()
    try:
        model.load()
    except Exception:
        print("Training model...")
        from src.data_loader import load_player_data
        df = load_player_data()
        model.train(df)

    live_players = []

    # Process live top scorers first (richest current match stats)
    if scorers_data and "scorers" in scorers_data:
        for item in scorers_data["scorers"]:
            p = item.get("player", {})
            t = item.get("team", {})
            p_id = p.get("id")

            p_meta = player_positions.get(p_id, {})
            pos = p_meta.get("position") or map_position(p.get("position") or "Offence")
            dob = p_meta.get("dateOfBirth") or p.get("dateOfBirth")
            age = calculate_age(dob)

            goals = int(item.get("goals") or 0)
            assists = int(item.get("assists") or 0)
            played = int(item.get("playedMatches") or 1)
            # Estimated minutes played so far this season
            minutes = played * 85

            t_name = t.get("shortName") or t.get("name") or p_meta.get("team", "Premier League")
            t_crest = t.get("crest") or p_meta.get("crest", "")

            stats = {
                "goals": goals,
                "assists": assists,
                "minutes_played": minutes,
                "age": age,
                "position": pos,
            }
            pred_val = model.predict_value(stats)
            explanation = model.explain_prediction(stats)

            live_players.append({
                "player_name": p.get("name"),
                "team": t_name,
                "team_crest": t_crest,
                "position": pos,
                "age": age,
                "goals": goals,
                "assists": assists,
                "minutes_played": minutes,
                "played_matches": played,
                "predicted_value_eur_m": round(pred_val, 1),
                "is_live_season": True,
                "season": "2024-25 (Live Matchday)",
                "explanation": explanation,
            })

    # Add key squad players across teams if not already in scorers
    existing_names = {lp["player_name"].lower() for lp in live_players}
    if teams_data and "teams" in teams_data:
        for t in teams_data["teams"]:
            t_name = t.get("shortName") or t.get("name")
            t_crest = t.get("crest", "")
            for sq in t.get("squad", [])[:6]:  # top squad members from each club
                name = sq.get("name")
                if name.lower() not in existing_names:
                    pos = map_position(sq.get("position"))
                    age = calculate_age(sq.get("dateOfBirth"))
                    # Baseline season stats for squad members
                    goals = 2 if pos == "Forward" else (1 if pos == "Midfielder" else 0)
                    assists = 2 if pos in ["Forward", "Midfielder"] else 0
                    minutes = 720
                    stats = {
                        "goals": goals,
                        "assists": assists,
                        "minutes_played": minutes,
                        "age": age,
                        "position": pos,
                    }
                    pred_val = model.predict_value(stats)
                    explanation = model.explain_prediction(stats)

                    live_players.append({
                        "player_name": name,
                        "team": t_name,
                        "team_crest": t_crest,
                        "position": pos,
                        "age": age,
                        "goals": goals,
                        "assists": assists,
                        "minutes_played": minutes,
                        "played_matches": 8,
                        "predicted_value_eur_m": round(pred_val, 1),
                        "is_live_season": True,
                        "season": "2024-25 (Live Squad)",
                        "explanation": explanation,
                    })
                    existing_names.add(name.lower())

    out_file = config.DATA_DIR / "live_premier_league_players.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(live_players, f, indent=2)

    print(f"Successfully processed and saved {len(live_players)} live Premier League players to {out_file.name}!")
    return live_players


if __name__ == "__main__":
    fetch_and_cache_live_data()
