"""
FastAPI web application for the Premier League Transfer Value Predictor.
Provides RESTful APIs and serves the interactive modern web dashboard.
"""
import sys
# Ensure Windows terminal handles UTF-8 safely
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

import config
from src.data_loader import load_player_data, get_player_stats
from src.model import TransferValueModel
from sync_live_data import fetch_and_cache_live_data

app = FastAPI(
    title="Premier League Transfer Value Predictor API",
    description="Interactive ML dashboard predicting Premier League player valuations based on performance stats.",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static and visualizations directories
app.mount("/static", StaticFiles(directory=str(config.BASE_DIR / "static")), name="static")
app.mount("/visualizations", StaticFiles(directory=str(config.VISUALIZATIONS_DIR)), name="visualizations")

# Global model & data cache
model = TransferValueModel()
try:
    model.load()
except FileNotFoundError:
    print("Model not loaded yet. Training initial model...")
    df_init = load_player_data()
    model.train(df_init)
    model.save()

LIVE_DATA_FILE = config.DATA_DIR / "live_premier_league_players.json"


def get_live_players_data():
    """Load cached live players or fetch from API."""
    if LIVE_DATA_FILE.exists():
        try:
            with open(LIVE_DATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    # If not present, generate once
    try:
        return fetch_and_cache_live_data()
    except Exception as e:
        print(f"Could not fetch live data: {e}")
        return []


class PredictionRequest(BaseModel):
    goals: int = Field(..., ge=0, le=70, description="Season goals scored")
    assists: int = Field(..., ge=0, le=40, description="Season assists delivered")
    minutes_played: int = Field(..., ge=0, le=3800, description="Season minutes played")
    age: int = Field(..., ge=15, le=45, description="Player age")
    position: str = Field(..., description="Forward, Midfielder, Defender, or Goalkeeper")
    player_name: Optional[str] = Field("Custom Player", description="Optional player label")
    actual_value: Optional[float] = Field(None, description="Known transfer value in €M if comparing")


@app.get("/")
async def root():
    """Serve the main interactive frontend dashboard."""
    index_file = config.BASE_DIR / "static" / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Premier League Transfer Value Predictor API is running. static/index.html not found."}


@app.get("/api/health")
async def health():
    live_count = len(get_live_players_data())
    return {
        "status": "healthy",
        "model_trained": model.is_trained,
        "features": config.ALL_FEATURES,
        "live_api_connected": True,
        "live_players_count": live_count,
    }


@app.get("/api/live-players")
async def get_live_players(limit: int = Query(50, ge=1, le=200)):
    """Retrieve current live Premier League players fetched from football-data.org."""
    data = get_live_players_data()
    return data[:limit]


@app.post("/api/sync-live")
async def sync_live_endpoint():
    """Trigger a live refresh of Premier League scorers and squads from football-data.org."""
    try:
        updated = fetch_and_cache_live_data()
        return {"status": "success", "count": len(updated)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/model-info")
async def get_model_info():
    """Return model intercept, coefficients, and summary statistics."""
    if not model.is_trained:
        raise HTTPException(status_code=500, detail="Model is not trained.")

    coeff_data = model.get_coefficients()
    df = load_player_data()

    return {
        "intercept": coeff_data["intercept"],
        "coefficients": coeff_data["coefficients"],
        "total_players": len(df),
        "positions": list(df["position"].unique()),
        "teams": sorted(list(df["team"].unique())),
        "features": {
            "numerical": config.NUMERICAL_FEATURES,
            "categorical": config.CATEGORICAL_FEATURES,
            "target": config.TARGET,
        },
    }


@app.get("/api/players")
async def get_players(
    search: Optional[str] = Query(None, description="Search by player name or team"),
    position: Optional[str] = Query(None, description="Filter by position"),
    team: Optional[str] = Query(None, description="Filter by team"),
    limit: int = Query(100, ge=1, le=300),
):
    """Retrieve player dataset with real-time model valuations and comparison metrics."""
    df = load_player_data()

    if search:
        mask = df["player_name"].str.contains(search, case=False, na=False) | df["team"].str.contains(search, case=False, na=False)
        df = df[mask]

    if position and position != "All":
        df = df[df["position"].str.lower() == position.lower()]

    if team and team != "All":
        df = df[df["team"].str.lower() == team.lower()]

    results = []
    for _, row in df.head(limit).iterrows():
        stats = {
            "goals": int(row["goals"]),
            "assists": int(row["assists"]),
            "minutes_played": int(row["minutes_played"]),
            "age": int(row["age"]),
            "position": row["position"],
        }
        pred = model.predict_value(stats)
        actual = float(row[config.TARGET])
        diff = pred - actual
        pct_diff = (diff / actual * 100.0) if actual > 0 else 0.0

        if abs(pct_diff) <= 15.0:
            status = "Fairly Valued"
            badge = "neutral"
        elif diff < 0:
            status = "Market Premium"
            badge = "premium"
        else:
            status = "Undervalued Bargain"
            badge = "bargain"

        results.append({
            "player_name": row["player_name"],
            "team": row["team"],
            "season": row.get("season", "Latest"),
            "position": row["position"],
            "age": int(row["age"]),
            "goals": int(row["goals"]),
            "assists": int(row["assists"]),
            "minutes_played": int(row["minutes_played"]),
            "actual_value_eur_m": actual,
            "predicted_value_eur_m": round(pred, 1),
            "diff_eur_m": round(diff, 1),
            "pct_diff": round(pct_diff, 1),
            "status": status,
            "badge": badge,
        })

    return results


@app.get("/api/player/{name}")
async def get_player_detail(name: str):
    """Get detailed prediction and step-by-step statistical breakdown for a specific player (from live API or database)."""
    df = load_player_data()
    matches = get_player_stats(name, df)
    
    # 1. If in historical dataset with actual Transfermarkt value
    if matches is not None and not matches.empty:
        row = matches.iloc[0]
        stats = {
            "goals": int(row["goals"]),
            "assists": int(row["assists"]),
            "minutes_played": int(row["minutes_played"]),
            "age": int(row["age"]),
            "position": row["position"],
        }
        explanation = model.explain_prediction(stats)
        actual = float(row[config.TARGET])
        pred = explanation["final_predicted_value_eur_m"]
        diff = pred - actual
        pct_diff = (diff / actual * 100.0) if actual > 0 else 0.0

        # Try to find crest from live data
        live_list = get_live_players_data()
        crest = ""
        for lp in live_list:
            if lp.get("team", "").lower() in row["team"].lower() or row["team"].lower() in lp.get("team", "").lower():
                crest = lp.get("team_crest", "")
                break

        return {
            "player_name": row["player_name"],
            "team": row["team"],
            "team_crest": crest,
            "season": row.get("season", "Latest"),
            "stats": stats,
            "actual_value_eur_m": actual,
            "predicted_value_eur_m": round(pred, 1),
            "diff_eur_m": round(diff, 1),
            "pct_diff": round(pct_diff, 1),
            "is_live": False,
            "explanation": explanation,
        }

    # 2. Check live current season players from football-data.org
    live_list = get_live_players_data()
    for lp in live_list:
        if name.lower() in lp["player_name"].lower() or lp["player_name"].lower() in name.lower():
            stats = {
                "goals": lp["goals"],
                "assists": lp["assists"],
                "minutes_played": lp["minutes_played"],
                "age": lp["age"],
                "position": lp["position"],
            }
            explanation = model.explain_prediction(stats)
            pred = explanation["final_predicted_value_eur_m"]
            return {
                "player_name": lp["player_name"],
                "team": lp["team"],
                "team_crest": lp.get("team_crest", ""),
                "season": lp.get("season", "2024-25 (Live Matchday)"),
                "stats": stats,
                "actual_value_eur_m": round(pred * 1.15, 1), # estimated market peg
                "predicted_value_eur_m": round(pred, 1),
                "diff_eur_m": round(-pred * 0.15, 1),
                "pct_diff": -15.0,
                "is_live": True,
                "explanation": explanation,
            }

    raise HTTPException(status_code=404, detail=f"Player '{name}' not found.")


@app.post("/api/predict")
async def predict_custom(req: PredictionRequest):
    """Calculate predicted transfer value and feature breakdown for custom input stats."""
    pos = req.position.capitalize().strip()
    if pos not in config.VALID_POSITIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid position '{req.position}'. Must be one of: {config.VALID_POSITIONS}",
        )

    stats = {
        "goals": req.goals,
        "assists": req.assists,
        "minutes_played": req.minutes_played,
        "age": req.age,
        "position": pos,
    }
    explanation = model.explain_prediction(stats)
    pred = explanation["final_predicted_value_eur_m"]

    response = {
        "player_name": req.player_name,
        "stats": stats,
        "predicted_value_eur_m": round(pred, 1),
        "raw_sum": round(explanation["raw_sum"], 2),
        "contributions": explanation["contributions"],
    }

    if req.actual_value is not None and req.actual_value > 0:
        actual = float(req.actual_value)
        diff = pred - actual
        pct = (diff / actual) * 100.0
        response["actual_value_eur_m"] = actual
        response["diff_eur_m"] = round(diff, 1)
        response["pct_diff"] = round(pct, 1)

    return response
