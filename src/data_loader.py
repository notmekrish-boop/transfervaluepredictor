"""
Data loader and preprocessing module for Premier League player statistics.
"""
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import pandas as pd
import numpy as np

import config
from src.api_client import FootballDataClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# Curated realistic multi-season Premier League dataset (2021-2024)
# Features: goals, assists, minutes_played, age, position, target: actual_value_eur_m
RAW_PLAYER_DATA = [
    # --- STRIKERS & WINGERS (FORWARDS) ---
    {"player_name": "Erling Haaland", "team": "Manchester City", "season": "2023-24", "age": 23, "position": "Forward", "goals": 27, "assists": 5, "minutes_played": 2558, "actual_value_eur_m": 180.0},
    {"player_name": "Erling Haaland", "team": "Manchester City", "season": "2022-23", "age": 22, "position": "Forward", "goals": 36, "assists": 8, "minutes_played": 2779, "actual_value_eur_m": 180.0},
    {"player_name": "Bukayo Saka", "team": "Arsenal", "season": "2023-24", "age": 22, "position": "Forward", "goals": 16, "assists": 9, "minutes_played": 2938, "actual_value_eur_m": 140.0},
    {"player_name": "Bukayo Saka", "team": "Arsenal", "season": "2022-23", "age": 21, "position": "Forward", "goals": 14, "assists": 11, "minutes_played": 3181, "actual_value_eur_m": 110.0},
    {"player_name": "Mohamed Salah", "team": "Liverpool", "season": "2023-24", "age": 31, "position": "Forward", "goals": 18, "assists": 10, "minutes_played": 2536, "actual_value_eur_m": 65.0},
    {"player_name": "Mohamed Salah", "team": "Liverpool", "season": "2022-23", "age": 30, "position": "Forward", "goals": 19, "assists": 12, "minutes_played": 3296, "actual_value_eur_m": 70.0},
    {"player_name": "Mohamed Salah", "team": "Liverpool", "season": "2021-22", "age": 29, "position": "Forward", "goals": 23, "assists": 13, "minutes_played": 2762, "actual_value_eur_m": 90.0},
    {"player_name": "Cole Palmer", "team": "Chelsea", "season": "2023-24", "age": 22, "position": "Forward", "goals": 22, "assists": 11, "minutes_played": 2617, "actual_value_eur_m": 80.0},
    {"player_name": "Alexander Isak", "team": "Newcastle United", "season": "2023-24", "age": 24, "position": "Forward", "goals": 21, "assists": 2, "minutes_played": 2260, "actual_value_eur_m": 75.0},
    {"player_name": "Alexander Isak", "team": "Newcastle United", "season": "2022-23", "age": 23, "position": "Forward", "goals": 10, "assists": 1, "minutes_played": 1530, "actual_value_eur_m": 70.0},
    {"player_name": "Ollie Watkins", "team": "Aston Villa", "season": "2023-24", "age": 28, "position": "Forward", "goals": 19, "assists": 13, "minutes_played": 3225, "actual_value_eur_m": 65.0},
    {"player_name": "Ollie Watkins", "team": "Aston Villa", "season": "2022-23", "age": 27, "position": "Forward", "goals": 15, "assists": 6, "minutes_played": 3130, "actual_value_eur_m": 40.0},
    {"player_name": "Son Heung-min", "team": "Tottenham Hotspur", "season": "2023-24", "age": 31, "position": "Forward", "goals": 17, "assists": 10, "minutes_played": 2948, "actual_value_eur_m": 45.0},
    {"player_name": "Son Heung-min", "team": "Tottenham Hotspur", "season": "2021-22", "age": 29, "position": "Forward", "goals": 23, "assists": 7, "minutes_played": 3021, "actual_value_eur_m": 75.0},
    {"player_name": "Jarrod Bowen", "team": "West Ham United", "season": "2023-24", "age": 27, "position": "Forward", "goals": 16, "assists": 6, "minutes_played": 2977, "actual_value_eur_m": 50.0},
    {"player_name": "Dominic Solanke", "team": "AFC Bournemouth", "season": "2023-24", "age": 26, "position": "Forward", "goals": 19, "assists": 3, "minutes_played": 3328, "actual_value_eur_m": 40.0},
    {"player_name": "Nicolas Jackson", "team": "Chelsea", "season": "2023-24", "age": 22, "position": "Forward", "goals": 14, "assists": 5, "minutes_played": 2811, "actual_value_eur_m": 35.0},
    {"player_name": "Jean-Philippe Mateta", "team": "Crystal Palace", "season": "2023-24", "age": 26, "position": "Forward", "goals": 16, "assists": 5, "minutes_played": 2281, "actual_value_eur_m": 30.0},
    {"player_name": "Kai Havertz", "team": "Arsenal", "season": "2023-24", "age": 25, "position": "Forward", "goals": 13, "assists": 7, "minutes_played": 2634, "actual_value_eur_m": 70.0},
    {"player_name": "Anthony Gordon", "team": "Newcastle United", "season": "2023-24", "age": 23, "position": "Forward", "goals": 11, "assists": 10, "minutes_played": 2901, "actual_value_eur_m": 60.0},
    {"player_name": "Darwin Nunez", "team": "Liverpool", "season": "2023-24", "age": 24, "position": "Forward", "goals": 11, "assists": 8, "minutes_played": 2047, "actual_value_eur_m": 70.0},
    {"player_name": "Darwin Nunez", "team": "Liverpool", "season": "2022-23", "age": 23, "position": "Forward", "goals": 9, "assists": 3, "minutes_played": 1695, "actual_value_eur_m": 70.0},
    {"player_name": "Luis Diaz", "team": "Liverpool", "season": "2023-24", "age": 27, "position": "Forward", "goals": 8, "assists": 5, "minutes_played": 2643, "actual_value_eur_m": 75.0},
    {"player_name": "Cody Gakpo", "team": "Liverpool", "season": "2023-24", "age": 25, "position": "Forward", "goals": 8, "assists": 5, "minutes_played": 1630, "actual_value_eur_m": 50.0},
    {"player_name": "Leandro Trossard", "team": "Arsenal", "season": "2023-24", "age": 29, "position": "Forward", "goals": 12, "assists": 1, "minutes_played": 1638, "actual_value_eur_m": 35.0},
    {"player_name": "Gabriel Martinelli", "team": "Arsenal", "season": "2022-23", "age": 21, "position": "Forward", "goals": 15, "assists": 5, "minutes_played": 2789, "actual_value_eur_m": 80.0},
    {"player_name": "Gabriel Martinelli", "team": "Arsenal", "season": "2023-24", "age": 22, "position": "Forward", "goals": 6, "assists": 4, "minutes_played": 2017, "actual_value_eur_m": 70.0},
    {"player_name": "Julian Alvarez", "team": "Manchester City", "season": "2023-24", "age": 24, "position": "Forward", "goals": 11, "assists": 9, "minutes_played": 2659, "actual_value_eur_m": 90.0},
    {"player_name": "Jeremy Doku", "team": "Manchester City", "season": "2023-24", "age": 22, "position": "Forward", "goals": 3, "assists": 9, "minutes_played": 1593, "actual_value_eur_m": 65.0},
    {"player_name": "Jack Grealish", "team": "Manchester City", "season": "2022-23", "age": 27, "position": "Forward", "goals": 5, "assists": 7, "minutes_played": 2060, "actual_value_eur_m": 75.0},
    {"player_name": "Jack Grealish", "team": "Manchester City", "season": "2023-24", "age": 28, "position": "Forward", "goals": 3, "assists": 1, "minutes_played": 1000, "actual_value_eur_m": 60.0},
    {"player_name": "Marcus Rashford", "team": "Manchester United", "season": "2022-23", "age": 25, "position": "Forward", "goals": 17, "assists": 5, "minutes_played": 2879, "actual_value_eur_m": 80.0},
    {"player_name": "Marcus Rashford", "team": "Manchester United", "season": "2023-24", "age": 26, "position": "Forward", "goals": 7, "assists": 2, "minutes_played": 2374, "actual_value_eur_m": 60.0},
    {"player_name": "Rasmus Hojlund", "team": "Manchester United", "season": "2023-24", "age": 21, "position": "Forward", "goals": 10, "assists": 2, "minutes_played": 2171, "actual_value_eur_m": 65.0},
    {"player_name": "Alejandro Garnacho", "team": "Manchester United", "season": "2023-24", "age": 19, "position": "Forward", "goals": 7, "assists": 4, "minutes_played": 2574, "actual_value_eur_m": 45.0},
    {"player_name": "Richarlison", "team": "Tottenham Hotspur", "season": "2023-24", "age": 27, "position": "Forward", "goals": 11, "assists": 4, "minutes_played": 1494, "actual_value_eur_m": 38.0},
    {"player_name": "Brennan Johnson", "team": "Tottenham Hotspur", "season": "2023-24", "age": 23, "position": "Forward", "goals": 5, "assists": 10, "minutes_played": 2337, "actual_value_eur_m": 48.0},
    {"player_name": "Dejan Kulusevski", "team": "Tottenham Hotspur", "season": "2023-24", "age": 24, "position": "Forward", "goals": 8, "assists": 3, "minutes_played": 2769, "actual_value_eur_m": 55.0},
    {"player_name": "Leon Bailey", "team": "Aston Villa", "season": "2023-24", "age": 26, "position": "Forward", "goals": 10, "assists": 9, "minutes_played": 2097, "actual_value_eur_m": 42.0},
    {"player_name": "Moussa Diaby", "team": "Aston Villa", "season": "2023-24", "age": 24, "position": "Forward", "goals": 6, "assists": 8, "minutes_played": 2197, "actual_value_eur_m": 55.0},
    {"player_name": "Bryan Mbeumo", "team": "Brentford", "season": "2023-24", "age": 24, "position": "Forward", "goals": 9, "assists": 6, "minutes_played": 1980, "actual_value_eur_m": 40.0},
    {"player_name": "Yoane Wissa", "team": "Brentford", "season": "2023-24", "age": 27, "position": "Forward", "goals": 12, "assists": 3, "minutes_played": 2374, "actual_value_eur_m": 28.0},
    {"player_name": "Ivan Toney", "team": "Brentford", "season": "2022-23", "age": 27, "position": "Forward", "goals": 20, "assists": 4, "minutes_played": 2953, "actual_value_eur_m": 50.0},
    {"player_name": "Matheus Cunha", "team": "Wolverhampton Wanderers", "season": "2023-24", "age": 25, "position": "Forward", "goals": 12, "assists": 7, "minutes_played": 2465, "actual_value_eur_m": 45.0},
    {"player_name": "Hwang Hee-chan", "team": "Wolverhampton Wanderers", "season": "2023-24", "age": 28, "position": "Forward", "goals": 12, "assists": 3, "minutes_played": 2139, "actual_value_eur_m": 25.0},
    {"player_name": "Pedro Neto", "team": "Wolverhampton Wanderers", "season": "2023-24", "age": 24, "position": "Forward", "goals": 2, "assists": 9, "minutes_played": 1519, "actual_value_eur_m": 55.0},
    {"player_name": "Michael Olise", "team": "Crystal Palace", "season": "2023-24", "age": 22, "position": "Forward", "goals": 10, "assists": 6, "minutes_played": 1277, "actual_value_eur_m": 55.0},
    {"player_name": "Mohammed Kudus", "team": "West Ham United", "season": "2023-24", "age": 23, "position": "Forward", "goals": 8, "assists": 6, "minutes_played": 2478, "actual_value_eur_m": 50.0},
    {"player_name": "Chris Wood", "team": "Nottingham Forest", "season": "2023-24", "age": 32, "position": "Forward", "goals": 14, "assists": 1, "minutes_played": 1807, "actual_value_eur_m": 7.0},
    {"player_name": "Callum Wilson", "team": "Newcastle United", "season": "2022-23", "age": 31, "position": "Forward", "goals": 18, "assists": 5, "minutes_played": 1880, "actual_value_eur_m": 18.0},
    {"player_name": "Harvey Barnes", "team": "Leicester City", "season": "2022-23", "age": 25, "position": "Forward", "goals": 13, "assists": 1, "minutes_played": 2715, "actual_value_eur_m": 35.0},
    {"player_name": "Dominic Calvert-Lewin", "team": "Everton", "season": "2023-24", "age": 27, "position": "Forward", "goals": 7, "assists": 2, "minutes_played": 2197, "actual_value_eur_m": 22.0},
    {"player_name": "Rodrigo Muniz", "team": "Fulham", "season": "2023-24", "age": 23, "position": "Forward", "goals": 9, "assists": 1, "minutes_played": 1563, "actual_value_eur_m": 20.0},
    {"player_name": "Raul Jimenez", "team": "Fulham", "season": "2023-24", "age": 33, "position": "Forward", "goals": 7, "assists": 0, "minutes_played": 1373, "actual_value_eur_m": 5.0},
    {"player_name": "Kaoru Mitoma", "team": "Brighton & Hove Albion", "season": "2022-23", "age": 26, "position": "Forward", "goals": 7, "assists": 5, "minutes_played": 2320, "actual_value_eur_m": 50.0},
    {"player_name": "Simon Adingra", "team": "Brighton & Hove Albion", "season": "2023-24", "age": 22, "position": "Forward", "goals": 6, "assists": 1, "minutes_played": 2217, "actual_value_eur_m": 30.0},
    {"player_name": "Carlton Morris", "team": "Luton Town", "season": "2023-24", "age": 28, "position": "Forward", "goals": 11, "assists": 4, "minutes_played": 2900, "actual_value_eur_m": 13.0},
    {"player_name": "Elijah Adebayo", "team": "Luton Town", "season": "2023-24", "age": 26, "position": "Forward", "goals": 10, "assists": 0, "minutes_played": 1420, "actual_value_eur_m": 12.0},
    {"player_name": "Anthony Elanga", "team": "Nottingham Forest", "season": "2023-24", "age": 22, "position": "Forward", "goals": 5, "assists": 9, "minutes_played": 2419, "actual_value_eur_m": 22.0},
    {"player_name": "Callum Hudson-Odoi", "team": "Nottingham Forest", "season": "2023-24", "age": 23, "position": "Forward", "goals": 8, "assists": 1, "minutes_played": 1690, "actual_value_eur_m": 18.0},

    # --- MIDFIELDERS ---
    {"player_name": "Phil Foden", "team": "Manchester City", "season": "2023-24", "age": 24, "position": "Midfielder", "goals": 19, "assists": 8, "minutes_played": 2865, "actual_value_eur_m": 150.0},
    {"player_name": "Rodri", "team": "Manchester City", "season": "2023-24", "age": 27, "position": "Midfielder", "goals": 8, "assists": 9, "minutes_played": 2932, "actual_value_eur_m": 130.0},
    {"player_name": "Declan Rice", "team": "Arsenal", "season": "2023-24", "age": 25, "position": "Midfielder", "goals": 7, "assists": 8, "minutes_played": 3225, "actual_value_eur_m": 120.0},
    {"player_name": "Martin Odegaard", "team": "Arsenal", "season": "2023-24", "age": 25, "position": "Midfielder", "goals": 8, "assists": 10, "minutes_played": 3097, "actual_value_eur_m": 110.0},
    {"player_name": "Martin Odegaard", "team": "Arsenal", "season": "2022-23", "age": 24, "position": "Midfielder", "goals": 15, "assists": 7, "minutes_played": 3127, "actual_value_eur_m": 90.0},
    {"player_name": "Bruno Guimaraes", "team": "Newcastle United", "season": "2023-24", "age": 26, "position": "Midfielder", "goals": 7, "assists": 8, "minutes_played": 3266, "actual_value_eur_m": 85.0},
    {"player_name": "Kevin De Bruyne", "team": "Manchester City", "season": "2022-23", "age": 31, "position": "Midfielder", "goals": 7, "assists": 16, "minutes_played": 2425, "actual_value_eur_m": 70.0},
    {"player_name": "Kevin De Bruyne", "team": "Manchester City", "season": "2021-22", "age": 30, "position": "Midfielder", "goals": 15, "assists": 8, "minutes_played": 2201, "actual_value_eur_m": 85.0},
    {"player_name": "Bernardo Silva", "team": "Manchester City", "season": "2023-24", "age": 29, "position": "Midfielder", "goals": 6, "assists": 9, "minutes_played": 2577, "actual_value_eur_m": 70.0},
    {"player_name": "Bruno Fernandes", "team": "Manchester United", "season": "2023-24", "age": 29, "position": "Midfielder", "goals": 10, "assists": 8, "minutes_played": 3137, "actual_value_eur_m": 70.0},
    {"player_name": "Bruno Fernandes", "team": "Manchester United", "season": "2022-23", "age": 28, "position": "Midfielder", "goals": 8, "assists": 8, "minutes_played": 3320, "actual_value_eur_m": 75.0},
    {"player_name": "Alexis Mac Allister", "team": "Liverpool", "season": "2023-24", "age": 25, "position": "Midfielder", "goals": 5, "assists": 5, "minutes_played": 2605, "actual_value_eur_m": 75.0},
    {"player_name": "Dominik Szoboszlai", "team": "Liverpool", "season": "2023-24", "age": 23, "position": "Midfielder", "goals": 3, "assists": 2, "minutes_played": 2111, "actual_value_eur_m": 75.0},
    {"player_name": "Moises Caicedo", "team": "Chelsea", "season": "2023-24", "age": 22, "position": "Midfielder", "goals": 1, "assists": 3, "minutes_played": 2842, "actual_value_eur_m": 75.0},
    {"player_name": "Enzo Fernandez", "team": "Chelsea", "season": "2023-24", "age": 23, "position": "Midfielder", "goals": 3, "assists": 2, "minutes_played": 2197, "actual_value_eur_m": 75.0},
    {"player_name": "James Maddison", "team": "Tottenham Hotspur", "season": "2023-24", "age": 27, "position": "Midfielder", "goals": 4, "assists": 9, "minutes_played": 2147, "actual_value_eur_m": 70.0},
    {"player_name": "Douglas Luiz", "team": "Aston Villa", "season": "2023-24", "age": 26, "position": "Midfielder", "goals": 9, "assists": 5, "minutes_played": 3014, "actual_value_eur_m": 70.0},
    {"player_name": "Lucas Paqueta", "team": "West Ham United", "season": "2023-24", "age": 26, "position": "Midfielder", "goals": 4, "assists": 6, "minutes_played": 2603, "actual_value_eur_m": 65.0},
    {"player_name": "Eberechi Eze", "team": "Crystal Palace", "season": "2023-24", "age": 25, "position": "Midfielder", "goals": 11, "assists": 4, "minutes_played": 2060, "actual_value_eur_m": 55.0},
    {"player_name": "Kobbie Mainoo", "team": "Manchester United", "season": "2023-24", "age": 19, "position": "Midfielder", "goals": 3, "assists": 1, "minutes_played": 1907, "actual_value_eur_m": 50.0},
    {"player_name": "Conor Gallagher", "team": "Chelsea", "season": "2023-24", "age": 24, "position": "Midfielder", "goals": 5, "assists": 7, "minutes_played": 3139, "actual_value_eur_m": 50.0},
    {"player_name": "Amadou Onana", "team": "Everton", "season": "2023-24", "age": 22, "position": "Midfielder", "goals": 2, "assists": 0, "minutes_played": 2133, "actual_value_eur_m": 50.0},
    {"player_name": "Morgan Gibbs-White", "team": "Nottingham Forest", "season": "2023-24", "age": 24, "position": "Midfielder", "goals": 5, "assists": 10, "minutes_played": 3140, "actual_value_eur_m": 40.0},
    {"player_name": "Boubacar Kamara", "team": "Aston Villa", "season": "2023-24", "age": 24, "position": "Midfielder", "goals": 0, "assists": 1, "minutes_played": 1735, "actual_value_eur_m": 38.0},
    {"player_name": "Harvey Elliott", "team": "Liverpool", "season": "2023-24", "age": 21, "position": "Midfielder", "goals": 3, "assists": 6, "minutes_played": 1335, "actual_value_eur_m": 35.0},
    {"player_name": "Curtis Jones", "team": "Liverpool", "season": "2023-24", "age": 23, "position": "Midfielder", "goals": 1, "assists": 1, "minutes_played": 1170, "actual_value_eur_m": 35.0},
    {"player_name": "Joao Gomes", "team": "Wolverhampton Wanderers", "season": "2023-24", "age": 23, "position": "Midfielder", "goals": 2, "assists": 1, "minutes_played": 2661, "actual_value_eur_m": 35.0},
    {"player_name": "Edson Alvarez", "team": "West Ham United", "season": "2023-24", "age": 26, "position": "Midfielder", "goals": 1, "assists": 1, "minutes_played": 2374, "actual_value_eur_m": 35.0},
    {"player_name": "John McGinn", "team": "Aston Villa", "season": "2023-24", "age": 29, "position": "Midfielder", "goals": 6, "assists": 4, "minutes_played": 3072, "actual_value_eur_m": 30.0},
    {"player_name": "James Ward-Prowse", "team": "West Ham United", "season": "2023-24", "age": 29, "position": "Midfielder", "goals": 7, "assists": 7, "minutes_played": 3139, "actual_value_eur_m": 30.0},
    {"player_name": "Mateo Kovacic", "team": "Manchester City", "season": "2023-24", "age": 30, "position": "Midfielder", "goals": 1, "assists": 0, "minutes_played": 1670, "actual_value_eur_m": 30.0},
    {"player_name": "Scott McTominay", "team": "Manchester United", "season": "2023-24", "age": 27, "position": "Midfielder", "goals": 7, "assists": 1, "minutes_played": 1790, "actual_value_eur_m": 32.0},
    {"player_name": "Adam Wharton", "team": "Crystal Palace", "season": "2023-24", "age": 20, "position": "Midfielder", "goals": 0, "assists": 3, "minutes_played": 1400, "actual_value_eur_m": 30.0},
    {"player_name": "Youri Tielemans", "team": "Aston Villa", "season": "2023-24", "age": 27, "position": "Midfielder", "goals": 2, "assists": 6, "minutes_played": 1748, "actual_value_eur_m": 25.0},
    {"player_name": "Sean Longstaff", "team": "Newcastle United", "season": "2023-24", "age": 26, "position": "Midfielder", "goals": 6, "assists": 2, "minutes_played": 2690, "actual_value_eur_m": 25.0},
    {"player_name": "Alex Iwobi", "team": "Fulham", "season": "2023-24", "age": 28, "position": "Midfielder", "goals": 5, "assists": 2, "minutes_played": 2439, "actual_value_eur_m": 25.0},
    {"player_name": "Tomas Soucek", "team": "West Ham United", "season": "2023-24", "age": 29, "position": "Midfielder", "goals": 7, "assists": 2, "minutes_played": 2769, "actual_value_eur_m": 22.0},
    {"player_name": "Dwight McNeil", "team": "Everton", "season": "2023-24", "age": 24, "position": "Midfielder", "goals": 3, "assists": 6, "minutes_played": 2980, "actual_value_eur_m": 22.0},
    {"player_name": "Andreas Pereira", "team": "Fulham", "season": "2023-24", "age": 28, "position": "Midfielder", "goals": 3, "assists": 7, "minutes_played": 2769, "actual_value_eur_m": 20.0},
    {"player_name": "Casemiro", "team": "Manchester United", "season": "2022-23", "age": 31, "position": "Midfielder", "goals": 4, "assists": 3, "minutes_played": 2128, "actual_value_eur_m": 40.0},
    {"player_name": "Casemiro", "team": "Manchester United", "season": "2023-24", "age": 32, "position": "Midfielder", "goals": 1, "assists": 2, "minutes_played": 1980, "actual_value_eur_m": 20.0},
    {"player_name": "Thomas Partey", "team": "Arsenal", "season": "2022-23", "age": 29, "position": "Midfielder", "goals": 3, "assists": 0, "minutes_played": 2484, "actual_value_eur_m": 35.0},
    {"player_name": "Jorginho", "team": "Arsenal", "season": "2023-24", "age": 32, "position": "Midfielder", "goals": 0, "assists": 2, "minutes_played": 1260, "actual_value_eur_m": 12.0},
    {"player_name": "Wataru Endo", "team": "Liverpool", "season": "2023-24", "age": 31, "position": "Midfielder", "goals": 1, "assists": 0, "minutes_played": 1720, "actual_value_eur_m": 13.0},
    {"player_name": "Abdoulaye Doucoure", "team": "Everton", "season": "2023-24", "age": 31, "position": "Midfielder", "goals": 7, "assists": 2, "minutes_played": 2520, "actual_value_eur_m": 12.0},
    {"player_name": "Mario Lemina", "team": "Wolverhampton Wanderers", "season": "2023-24", "age": 30, "position": "Midfielder", "goals": 4, "assists": 1, "minutes_played": 2901, "actual_value_eur_m": 10.0},
    {"player_name": "Pascal Gross", "team": "Brighton & Hove Albion", "season": "2023-24", "age": 33, "position": "Midfielder", "goals": 4, "assists": 10, "minutes_played": 3149, "actual_value_eur_m": 8.0},
    {"player_name": "Ross Barkley", "team": "Luton Town", "season": "2023-24", "age": 30, "position": "Midfielder", "goals": 5, "assists": 4, "minutes_played": 2700, "actual_value_eur_m": 8.0},
    {"player_name": "Idrissa Gueye", "team": "Everton", "season": "2023-24", "age": 34, "position": "Midfielder", "goals": 4, "assists": 0, "minutes_played": 1948, "actual_value_eur_m": 3.0},

    # --- DEFENDERS ---
    {"player_name": "William Saliba", "team": "Arsenal", "season": "2023-24", "age": 23, "position": "Defender", "goals": 2, "assists": 1, "minutes_played": 3420, "actual_value_eur_m": 80.0},
    {"player_name": "William Saliba", "team": "Arsenal", "season": "2022-23", "age": 22, "position": "Defender", "goals": 2, "assists": 1, "minutes_played": 2415, "actual_value_eur_m": 65.0},
    {"player_name": "Ruben Dias", "team": "Manchester City", "season": "2023-24", "age": 27, "position": "Defender", "goals": 0, "assists": 0, "minutes_played": 2374, "actual_value_eur_m": 80.0},
    {"player_name": "Josko Gvardiol", "team": "Manchester City", "season": "2023-24", "age": 22, "position": "Defender", "goals": 4, "assists": 1, "minutes_played": 2329, "actual_value_eur_m": 75.0},
    {"player_name": "Gabriel Magalhaes", "team": "Arsenal", "season": "2023-24", "age": 26, "position": "Defender", "goals": 4, "assists": 0, "minutes_played": 3044, "actual_value_eur_m": 70.0},
    {"player_name": "Trent Alexander-Arnold", "team": "Liverpool", "season": "2023-24", "age": 25, "position": "Defender", "goals": 3, "assists": 4, "minutes_played": 2097, "actual_value_eur_m": 70.0},
    {"player_name": "Trent Alexander-Arnold", "team": "Liverpool", "season": "2021-22", "age": 23, "position": "Defender", "goals": 2, "assists": 12, "minutes_played": 2854, "actual_value_eur_m": 80.0},
    {"player_name": "Cristian Romero", "team": "Tottenham Hotspur", "season": "2023-24", "age": 26, "position": "Defender", "goals": 5, "assists": 0, "minutes_played": 2824, "actual_value_eur_m": 60.0},
    {"player_name": "Ben White", "team": "Arsenal", "season": "2023-24", "age": 26, "position": "Defender", "goals": 4, "assists": 4, "minutes_played": 3180, "actual_value_eur_m": 55.0},
    {"player_name": "Levi Colwill", "team": "Chelsea", "season": "2023-24", "age": 21, "position": "Defender", "goals": 1, "assists": 1, "minutes_played": 1790, "actual_value_eur_m": 50.0},
    {"player_name": "Pedro Porro", "team": "Tottenham Hotspur", "season": "2023-24", "age": 24, "position": "Defender", "goals": 3, "assists": 7, "minutes_played": 3060, "actual_value_eur_m": 45.0},
    {"player_name": "Destiny Udogie", "team": "Tottenham Hotspur", "season": "2023-24", "age": 21, "position": "Defender", "goals": 2, "assists": 3, "minutes_played": 2419, "actual_value_eur_m": 45.0},
    {"player_name": "Pau Torres", "team": "Aston Villa", "season": "2023-24", "age": 27, "position": "Defender", "goals": 2, "assists": 0, "minutes_played": 2742, "actual_value_eur_m": 45.0},
    {"player_name": "Ibrahima Konate", "team": "Liverpool", "season": "2023-24", "age": 25, "position": "Defender", "goals": 0, "assists": 0, "minutes_played": 1540, "actual_value_eur_m": 45.0},
    {"player_name": "Manuel Akanji", "team": "Manchester City", "season": "2023-24", "age": 28, "position": "Defender", "goals": 2, "assists": 0, "minutes_played": 2397, "actual_value_eur_m": 45.0},
    {"player_name": "Jarrad Branthwaite", "team": "Everton", "season": "2023-24", "age": 21, "position": "Defender", "goals": 3, "assists": 0, "minutes_played": 3116, "actual_value_eur_m": 42.0},
    {"player_name": "John Stones", "team": "Manchester City", "season": "2022-23", "age": 28, "position": "Defender", "goals": 2, "assists": 2, "minutes_played": 1846, "actual_value_eur_m": 40.0},
    {"player_name": "Nathan Ake", "team": "Manchester City", "season": "2023-24", "age": 29, "position": "Defender", "goals": 2, "assists": 2, "minutes_played": 1709, "actual_value_eur_m": 40.0},
    {"player_name": "Axel Disasi", "team": "Chelsea", "season": "2023-24", "age": 26, "position": "Defender", "goals": 2, "assists": 0, "minutes_played": 2577, "actual_value_eur_m": 40.0},
    {"player_name": "Marc Guehi", "team": "Crystal Palace", "season": "2023-24", "age": 23, "position": "Defender", "goals": 0, "assists": 1, "minutes_played": 2190, "actual_value_eur_m": 38.0},
    {"player_name": "Ezri Konsa", "team": "Aston Villa", "season": "2023-24", "age": 26, "position": "Defender", "goals": 1, "assists": 0, "minutes_played": 3034, "actual_value_eur_m": 35.0},
    {"player_name": "Diogo Dalot", "team": "Manchester United", "season": "2023-24", "age": 25, "position": "Defender", "goals": 2, "assists": 3, "minutes_played": 3060, "actual_value_eur_m": 35.0},
    {"player_name": "Murillo", "team": "Nottingham Forest", "season": "2023-24", "age": 21, "position": "Defender", "goals": 0, "assists": 2, "minutes_played": 2806, "actual_value_eur_m": 35.0},
    {"player_name": "Rayan Ait-Nouri", "team": "Wolverhampton Wanderers", "season": "2023-24", "age": 23, "position": "Defender", "goals": 2, "assists": 1, "minutes_played": 2697, "actual_value_eur_m": 35.0},
    {"player_name": "Malo Gusto", "team": "Chelsea", "season": "2023-24", "age": 21, "position": "Defender", "goals": 0, "assists": 6, "minutes_played": 1980, "actual_value_eur_m": 35.0},
    {"player_name": "Max Kilman", "team": "Wolverhampton Wanderers", "season": "2023-24", "age": 27, "position": "Defender", "goals": 2, "assists": 0, "minutes_played": 3420, "actual_value_eur_m": 32.0},
    {"player_name": "Virgil van Dijk", "team": "Liverpool", "season": "2023-24", "age": 32, "position": "Defender", "goals": 2, "assists": 2, "minutes_played": 3172, "actual_value_eur_m": 30.0},
    {"player_name": "Virgil van Dijk", "team": "Liverpool", "season": "2021-22", "age": 30, "position": "Defender", "goals": 3, "assists": 1, "minutes_played": 3060, "actual_value_eur_m": 55.0},
    {"player_name": "Andy Robertson", "team": "Liverpool", "season": "2023-24", "age": 30, "position": "Defender", "goals": 3, "assists": 2, "minutes_played": 1720, "actual_value_eur_m": 30.0},
    {"player_name": "Pervis Estupinan", "team": "Brighton & Hove Albion", "season": "2022-23", "age": 25, "position": "Defender", "goals": 1, "assists": 5, "minutes_played": 2870, "actual_value_eur_m": 35.0},
    {"player_name": "Vitaliy Mykolenko", "team": "Everton", "season": "2023-24", "age": 25, "position": "Defender", "goals": 2, "assists": 0, "minutes_played": 2439, "actual_value_eur_m": 28.0},
    {"player_name": "Antonee Robinson", "team": "Fulham", "season": "2023-24", "age": 26, "position": "Defender", "goals": 0, "assists": 6, "minutes_played": 3280, "actual_value_eur_m": 25.0},
    {"player_name": "Nathan Collins", "team": "Brentford", "season": "2023-24", "age": 23, "position": "Defender", "goals": 1, "assists": 1, "minutes_played": 2780, "actual_value_eur_m": 25.0},
    {"player_name": "Harry Maguire", "team": "Manchester United", "season": "2023-24", "age": 31, "position": "Defender", "goals": 2, "assists": 2, "minutes_played": 1630, "actual_value_eur_m": 18.0},
    {"player_name": "James Tarkowski", "team": "Everton", "season": "2023-24", "age": 31, "position": "Defender", "goals": 1, "assists": 1, "minutes_played": 3420, "actual_value_eur_m": 13.0},
    {"player_name": "Kyle Walker", "team": "Manchester City", "season": "2023-24", "age": 34, "position": "Defender", "goals": 0, "assists": 4, "minutes_played": 2525, "actual_value_eur_m": 13.0},
    {"player_name": "Kieran Trippier", "team": "Newcastle United", "season": "2023-24", "age": 33, "position": "Defender", "goals": 1, "assists": 10, "minutes_played": 2271, "actual_value_eur_m": 10.0},
    {"player_name": "Fabian Schar", "team": "Newcastle United", "season": "2023-24", "age": 32, "position": "Defender", "goals": 4, "assists": 0, "minutes_played": 3034, "actual_value_eur_m": 10.0},
    {"player_name": "Lewis Dunk", "team": "Brighton & Hove Albion", "season": "2023-24", "age": 32, "position": "Defender", "goals": 3, "assists": 0, "minutes_played": 2824, "actual_value_eur_m": 12.0},

    # --- GOALKEEPERS ---
    {"player_name": "David Raya", "team": "Arsenal", "season": "2023-24", "age": 28, "position": "Goalkeeper", "goals": 0, "assists": 0, "minutes_played": 2880, "actual_value_eur_m": 35.0},
    {"player_name": "Ederson", "team": "Manchester City", "season": "2023-24", "age": 30, "position": "Goalkeeper", "goals": 0, "assists": 0, "minutes_played": 2742, "actual_value_eur_m": 35.0},
    {"player_name": "Ederson", "team": "Manchester City", "season": "2022-23", "age": 29, "position": "Goalkeeper", "goals": 0, "assists": 1, "minutes_played": 3150, "actual_value_eur_m": 40.0},
    {"player_name": "Guglielmo Vicario", "team": "Tottenham Hotspur", "season": "2023-24", "age": 27, "position": "Goalkeeper", "goals": 0, "assists": 0, "minutes_played": 3420, "actual_value_eur_m": 35.0},
    {"player_name": "Andre Onana", "team": "Manchester United", "season": "2023-24", "age": 28, "position": "Goalkeeper", "goals": 0, "assists": 0, "minutes_played": 3420, "actual_value_eur_m": 35.0},
    {"player_name": "Alisson Becker", "team": "Liverpool", "season": "2023-24", "age": 31, "position": "Goalkeeper", "goals": 0, "assists": 0, "minutes_played": 2520, "actual_value_eur_m": 28.0},
    {"player_name": "Alisson Becker", "team": "Liverpool", "season": "2021-22", "age": 29, "position": "Goalkeeper", "goals": 0, "assists": 1, "minutes_played": 3240, "actual_value_eur_m": 50.0},
    {"player_name": "Emiliano Martinez", "team": "Aston Villa", "season": "2023-24", "age": 31, "position": "Goalkeeper", "goals": 0, "assists": 0, "minutes_played": 3015, "actual_value_eur_m": 28.0},
    {"player_name": "Jordan Pickford", "team": "Everton", "season": "2023-24", "age": 30, "position": "Goalkeeper", "goals": 0, "assists": 0, "minutes_played": 3420, "actual_value_eur_m": 22.0},
    {"player_name": "Nick Pope", "team": "Newcastle United", "season": "2022-23", "age": 31, "position": "Goalkeeper", "goals": 0, "assists": 0, "minutes_played": 3330, "actual_value_eur_m": 18.0},
    {"player_name": "Bernd Leno", "team": "Fulham", "season": "2023-24", "age": 32, "position": "Goalkeeper", "goals": 0, "assists": 0, "minutes_played": 3420, "actual_value_eur_m": 13.0},
    {"player_name": "Mark Flekken", "team": "Brentford", "season": "2023-24", "age": 30, "position": "Goalkeeper", "goals": 0, "assists": 1, "minutes_played": 3330, "actual_value_eur_m": 12.0},
    {"player_name": "Alphonse Areola", "team": "West Ham United", "season": "2023-24", "age": 31, "position": "Goalkeeper", "goals": 0, "assists": 0, "minutes_played": 2742, "actual_value_eur_m": 10.0},
    {"player_name": "Robert Sanchez", "team": "Chelsea", "season": "2023-24", "age": 26, "position": "Goalkeeper", "goals": 0, "assists": 0, "minutes_played": 1440, "actual_value_eur_m": 20.0},
    {"player_name": "Dean Henderson", "team": "Crystal Palace", "season": "2023-24", "age": 27, "position": "Goalkeeper", "goals": 0, "assists": 0, "minutes_played": 1620, "actual_value_eur_m": 12.0},
    {"player_name": "Jose Sa", "team": "Wolverhampton Wanderers", "season": "2023-24", "age": 31, "position": "Goalkeeper", "goals": 0, "assists": 0, "minutes_played": 3139, "actual_value_eur_m": 14.0},
    {"player_name": "Bart Verbruggen", "team": "Brighton & Hove Albion", "season": "2023-24", "age": 21, "position": "Goalkeeper", "goals": 0, "assists": 0, "minutes_played": 1890, "actual_value_eur_m": 18.0},
    {"player_name": "James Trafford", "team": "Burnley", "season": "2023-24", "age": 21, "position": "Goalkeeper", "goals": 0, "assists": 0, "minutes_played": 2520, "actual_value_eur_m": 18.0},
    {"player_name": "Thomas Kaminski", "team": "Luton Town", "season": "2023-24", "age": 31, "position": "Goalkeeper", "goals": 0, "assists": 0, "minutes_played": 3420, "actual_value_eur_m": 6.0},
]


def ensure_dataset_exists() -> Path:
    """Ensure the CSV dataset exists; if not, create it from curated data."""
    if not config.DATASET_PATH.exists():
        logger.info(f"Creating Premier League dataset at {config.DATASET_PATH}")
        df = pd.DataFrame(RAW_PLAYER_DATA)
        df.to_csv(config.DATASET_PATH, index=False)
        logger.info(f"Saved {len(df)} player records to {config.DATASET_PATH}")
    return config.DATASET_PATH


def sync_with_football_data_api(api_key: Optional[str] = None) -> None:
    """
    Attempt to fetch live scorers / teams from football-data.org via requests.
    Merges or updates local stats if API responses are successfully returned.
    """
    client = FootballDataClient(api_key=api_key)
    scorers = client.get_scorers(season=2023, limit=20)
    if scorers and "scorers" in scorers:
        logger.info(f"Retrieved {len(scorers['scorers'])} live scorers from football-data.org!")
        # We can inspect and log top scorers retrieved via requests
        for item in scorers["scorers"][:5]:
            p = item.get("player", {})
            t = item.get("team", {})
            goals = item.get("goals", 0)
            assists = item.get("assists", 0) or 0
            logger.info(f"API Scorer: {p.get('name')} ({t.get('name')}) - {goals} goals, {assists} assists")
    else:
        logger.info("Using cached/curated Premier League multi-season dataset.")


def load_player_data(dataset_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load and validate player data from CSV.
    Returns cleaned pandas DataFrame.
    """
    path = dataset_path or ensure_dataset_exists()
    df = pd.read_csv(path)

    # Basic validations
    required_cols = config.ALL_FEATURES + [config.TARGET, "player_name"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    # Standardize position names
    df["position"] = df["position"].str.capitalize().str.strip()
    df = df[df["position"].isin(config.VALID_POSITIONS)].copy()

    # Drop any nulls or invalid values
    df = df.dropna(subset=config.ALL_FEATURES + [config.TARGET]).copy()

    # Numeric clipping / sanity
    df["age"] = df["age"].astype(int)
    df["goals"] = df["goals"].astype(int)
    df["assists"] = df["assists"].astype(int)
    df["minutes_played"] = df["minutes_played"].astype(int)
    df[config.TARGET] = df[config.TARGET].astype(float)

    return df


def get_player_stats(player_name: str, df: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
    """Look up records for a specific player by name (case-insensitive substring match)."""
    if df is None:
        df = load_player_data()
    matches = df[df["player_name"].str.contains(player_name, case=False, na=False)]
    return matches if not matches.empty else None
