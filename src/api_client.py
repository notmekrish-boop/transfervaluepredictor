"""
API Client for football-data.org using the requests library.
Supports fetching Premier League scorers, teams, and squad statistics with caching.
"""
import json
import logging
import os
import requests
from pathlib import Path
from typing import Dict, Any, Optional

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class FootballDataClient:
    """Client for querying the football-data.org v4 REST API."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.FOOTBALL_DATA_API_KEY
        self.base_url = config.FOOTBALL_DATA_BASE_URL
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"X-Auth-Token": self.api_key})
        self.session.headers.update({"User-Agent": "TransferValuePredictor/1.0"})

    def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None, cache_filename: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Internal GET helper with disk caching and rate limit / auth handling.
        """
        # Check cache first
        cache_path = None
        if cache_filename:
            cache_path = config.RAW_DATA_DIR / cache_filename
            if cache_path.exists():
                logger.info(f"Loading cached response from {cache_path.name}")
                try:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to read cache file {cache_path}: {e}")

        if not self.api_key:
            logger.warning(
                "No FOOTBALL_DATA_API_KEY provided. football-data.org v4 requires a free API token. "
                "Register at https://www.football-data.org/client/register to use live API fetching. "
                "Using local cached/curated dataset instead."
            )
            return None

        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        logger.info(f"Making requests.get to {url} with params {params}")

        try:
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if cache_path:
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
                    logger.info(f"Saved response to cache: {cache_path.name}")
                return data
            elif response.status_code == 403:
                logger.error(
                    "Received 403 Forbidden from football-data.org. "
                    "Your API key may be invalid or does not have access to this resource."
                )
            elif response.status_code == 429:
                logger.error("Received 429 Rate Limited. Free tier is limited to 10 requests per minute.")
            else:
                logger.error(f"HTTP Error {response.status_code}: {response.text}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Network request to football-data.org failed: {e}")

        return None

    def get_scorers(self, competition: str = config.PREMIER_LEAGUE_CODE, season: Optional[int] = None, limit: int = 50) -> Optional[Dict[str, Any]]:
        """
        Fetch top scorers for a competition.
        Endpoint: /v4/competitions/{competition}/scorers
        """
        params = {"limit": limit}
        if season:
            params["season"] = season
        cache_file = f"scorers_{competition}_{season or 'current'}.json"
        return self._get(f"competitions/{competition}/scorers", params=params, cache_filename=cache_file)

    def get_teams(self, competition: str = config.PREMIER_LEAGUE_CODE, season: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Fetch all teams and squads in the Premier League.
        Endpoint: /v4/competitions/{competition}/teams
        """
        params = {}
        if season:
            params["season"] = season
        cache_file = f"teams_{competition}_{season or 'current'}.json"
        return self._get(f"competitions/{competition}/teams", params=params, cache_filename=cache_file)

    def get_team_squad(self, team_id: int) -> Optional[Dict[str, Any]]:
        """
        Fetch team details including current squad players.
        Endpoint: /v4/teams/{id}
        """
        cache_file = f"team_{team_id}.json"
        return self._get(f"teams/{team_id}", cache_filename=cache_file)
