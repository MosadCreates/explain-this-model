import logging
import os
import random
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI

_SEED = 42
random.seed(_SEED)
np.random.seed(_SEED)
torch.manual_seed(_SEED)
torch.cuda.manual_seed_all(_SEED)
from fastapi.middleware.cors import CORSMiddleware

from src.api.router import router as api_router
from src.config import get_config, set_config_path
from src.database import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    base_dir = os.path.dirname(os.path.dirname(__file__))
    config_path = os.path.join(base_dir, "configs", "default.yaml")
    if os.path.exists(config_path):
        set_config_path(config_path)
    else:
        logger.warning("Config file not found at %s, using defaults", config_path)

    get_config()
    init_db()
    logger.info("Application startup complete")
    yield
    logger.info("Application shutdown")


app = FastAPI(
    title="Explain This Model",
    description="Interpretability-as-a-Service — analyse neurons and attention heads in HuggingFace transformer models",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/")
def root():
    return {
        "service": "Explain This Model",
        "version": "0.1.0",
        "docs": "/docs",
    }
