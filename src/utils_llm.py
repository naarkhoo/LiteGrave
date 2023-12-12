"""Utilities for the language model."""
import ast
import os
import pickle
from typing import Any, Dict

import pandas as pd
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import GrobidParser
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from transformers import GPT2TokenizerFast

# Load the configuration
cfg = OmegaConf.load("conf/config.yaml")

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")


def count_tokens(text: str) -> int:
    """Count the number of tokens in a text."""
    return len(tokenizer.encode(text))


def index_pdf(pdf_file: str, cfg: DictConfig, write: bool = False) -> dict:
    """Index a PDF file using gorbin."""
    # check if pdf_file exist
    if not os.path.exists(pdf_file):
        raise ValueError(f"{pdf_file} does not exist")

    logger.info(f"crunching {pdf_file} through gorbin ...")
    logger.info(f"my glob pattern:{pdf_file.split('/')[-1].replace('.pdf', '')}*")
    logger.info(f"my filepath: {cfg.data.pdf_path}")
    loader = GenericLoader.from_filesystem(
        cfg.data.pdf_path + "/",
        glob=f"{pdf_file.split('/')[-1].replace('.pdf', '')}*",
        suffixes=[".pdf"],
        parser=GrobidParser(segment_sentences=True),
    )
    documents = loader.load()

    logger.info(f"parsed {len(documents)} docs using gorbin")

    data = []
    unique_texts = set()

    if len(documents) == 0:
        logger.error(f"Failed to grobid file: {pdf_file}")

    for t in documents:
        text = t.page_content
        file_path = t.metadata["file_path"]

        unique_key = (text, file_path)

        if unique_key not in unique_texts:
            unique_texts.add(unique_key)
            d = {}
            d["text"] = text
            d["page"] = ast.literal_eval(t.metadata["pages"])[
                0
            ]  # take only the first page
            d["section"] = t.metadata["section_title"]
            d["paper"] = t.metadata["paper_title"]
            d["file"] = file_path
            data.append(d)

    filename = t.metadata["file_path"]
    df = pd.DataFrame(data).drop(columns=["paper", "file"])

    # data_section: Dict[str, Union[List[str], str]] = {}
    # data_page: Dict[Any, Union[List[str], str]] = {}

    data_section: Dict[Any, Any] = {}
    data_page: Dict[Any, Any] = {}

    unique_texts = set()

    # label each identified sentence
    for t in documents:
        text = t.page_content
        file_path = t.metadata["file_path"]
        section = t.metadata["section_title"]
        page_number = ast.literal_eval(t.metadata["pages"])[
            0
        ]  # take only the first page

        unique_json_key = (text, file_path, section)

        if unique_json_key not in unique_texts:
            unique_texts.add(unique_json_key)  # type: ignore[arg-type]

            # Check if the section already exists in the data dictionary
            if section not in data_section:
                data_section[section] = []
            data_section[section].append(text)

            if page_number not in data_page:
                data_page[page_number] = []
            data_page[page_number].append(text)

    # Concatenating texts within each section
    for section, texts in data_section.items():
        data_section[section] = " ".join(texts)

    for page_number, texts in data_page.items():
        data_page[page_number] = " ".join(texts)

    if not os.path.exists(cfg.data.pdf_preprocessed_path):
        os.makedirs(cfg.data.pdf_preprocessed_path)

    filename = file_path.replace(
        cfg.data.pdf_path, cfg.data.pdf_preprocessed_path
    ).replace(".pdf", ".pkl")

    # Combine the variables into a dictionary
    data_to_save = {
        "data_section": data_section,
        "data_page": data_page,
        "dataframe": df,
    }

    if write:
        # Writing the combined data to a pickle file
        with open(filename, "wb") as file:
            pickle.dump(data_to_save, file)

        logger.info(f"Data saved to {filename}")

    return data_to_save
