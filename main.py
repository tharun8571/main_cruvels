# Entry point for Railway/Railpack auto-detection
# Railpack detects FastAPI apps from root main.py
from src.api.main import app  # noqa: F401

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
