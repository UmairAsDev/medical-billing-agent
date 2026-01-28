import sys
import uvicorn
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.app.api import router as superbill_router

app = FastAPI()

app.include_router(superbill_router, prefix="/api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def main():
    uvicorn.run("main:app", host="127.0.0.1", port=8005, reload=True, log_level="info")
    
    
    
if __name__ == "__main__":
    main()


