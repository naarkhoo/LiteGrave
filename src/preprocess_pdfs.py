"""Preprocess raw PDF files using index_pdf."""
from glob import glob

import inflect
from loguru import logger
from omegaconf import OmegaConf

from utils_llm import index_pdf

cfg = OmegaConf.load("conf/config.yaml")

all_pdf_files = glob(cfg.data.pdf_path + "/*.pdf")
logger.info(f"Found {len(all_pdf_files)} PDF files")
p = inflect.engine()

pubmed_id_list = [
    20213684,
    21169425,
    22296880,
    22347458,
    22808101,
    24324620,
    25965668,
    26061167,
    26083373,
    26295583,
    31708475,
    32900939,
    33406425,
]

all_pdf_files = [
    item
    for item in all_pdf_files
    if int(item.split("/")[-1].split(".")[0]) in pubmed_id_list
]

logger.info(f"Selected {len(all_pdf_files)} PDF files")

for pdf_file in all_pdf_files:
    logger.info(f"Indexing {pdf_file}")
    output = index_pdf(pdf_file, cfg, p, write=True)

logger.success("Done!")
