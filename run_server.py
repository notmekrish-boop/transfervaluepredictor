"""
Server launcher for Premier League Transfer Value Predictor web app.
Launches the FastAPI backend and serves the interactive dashboard.
"""
import sys
# Ensure Windows terminal handles UTF-8 safely
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os
import uvicorn
import config
from src.data_loader import ensure_dataset_exists
from src.model import TransferValueModel


def start():
    print("=" * 75)
    print("       PREMIER LEAGUE TRANSFER VALUE PREDICTOR - WEB DASHBOARD")
    print("=" * 75)

    # Check dataset and model
    ensure_dataset_exists()
    if not config.MODEL_PATH.exists():
        print("[*] Model file not found. Running training pipeline...")
        from train import main as run_train
        run_train()

    host = "127.0.0.1"
    port = 8000

    # Auto-detect if port 8000 is already in use
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex((host, port)) == 0:
            port = 8001

    print(f"\n🚀 Server starting at: http://{host}:{port}")
    print(f"📖 Interactive Dashboard: http://{host}:{port}")
    print(f"📚 Swagger API Docs:       http://{host}:{port}/docs")
    print("\nPress Ctrl+C to stop the server.\n")

    uvicorn.run("app:app", host=host, port=port, reload=False, log_level="info")


if __name__ == "__main__":
    start()
