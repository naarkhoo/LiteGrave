"""Test utils_llm.index_pdf() function."""
from glob import glob

import inflect
from loguru import logger
from omegaconf import OmegaConf

from utils_llm import index_pdf

cfg = OmegaConf.load("conf/config.yaml")

all_pdf_files = glob(cfg.data.pdf_path + "/*.pdf")
logger.info(f"Found {len(all_pdf_files)} PDF files")
p = inflect.engine()

for pdf_file in all_pdf_files[0:5]:
    logger.info(f"Indexing {pdf_file}")
    output = index_pdf(pdf_file, cfg, p, write=True)

logger.success("Done!")
