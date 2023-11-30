"""Module to index columns in the dataset."""
import pandas as pd
from loguru import logger
from omegaconf import OmegaConf

from utils import create_embeddings

# Load the configuration
cfg = OmegaConf.load("conf/config.yaml")

FILE_PATH = cfg.data.path
INDEXED_FILE_PATH = cfg.data.indexed_path

df = pd.read_csv(cfg.data.path, compression="gzip", header=0)
logger.info(f"Loaded {len(df)} rows from {cfg.data.path}")
logger.info(f"columns: {df.columns}")

logger.info("Creating embeddings for experiment time")
df = create_embeddings(df, ["experiment time"])

logger.info("Creating embeddings for device")
df = create_embeddings(df, ["device"])

print(df.head(5))

logger.info(f"Saving indexed file to {INDEXED_FILE_PATH}")
df.to_csv(INDEXED_FILE_PATH, compression="gzip", index=False)

logger.success("Done!")
