"""Utility functions for the project."""
from typing import Any, List, Union

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from loguru import logger
from omegaconf import OmegaConf
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the configuration
cfg = OmegaConf.load("conf/config.yaml")

# Load environment variables from .env file
load_dotenv(dotenv_path=cfg.credentials.path)


def exact_match(df: pd.DataFrame, colname: str, term: str) -> pd.DataFrame:
    """Search for an exact match in a DataFrame column."""
    if len(term) < 3:
        return df

    # Check if the column exists in the DataFrame
    if colname not in df.columns:
        raise ValueError(f"Column '{colname}' does not exist in DataFrame.")

    # Construct the query string
    query_str = (
        f"`{colname}` == '{term}'"
        if isinstance(term, str)
        else f"`{colname}` == {term}"
    )

    return df.query(query_str)


def exact_match_llm(df: pd.DataFrame, agent: Any, cell_type: str) -> str:
    """Search for cell type in the database using LLM."""
    # cell_type = 'Peripheral blood lymphocytes (PBL)'
    logger.info(f"Searching for cell type: {cell_type}")
    response = agent.run(
        f"which rows have 'cell type' equal to {cell_type}"
        f"return the row ids as a list of integers; if no rows match"
        f", return an empty list"
    )
    return response


def semantic_search_llm(
    df: pd.DataFrame, agent: Any, columns: List[str], query: str
) -> str:
    """Search for cell type in the database using LLM."""
    # cell_type = 'Peripheral blood lymphocytes (PBL)'
    logger.info(f"Searching in columns: {columns}")
    response = agent.run(
        f"which rows of columns {','.join(columns)} is equal to {query}"
        f"return the row ids as a list of integers; if no rows match"
        f", return an empty list"
    )
    return response


def create_embeddings(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Create embeddings for a list of columns in a DataFrame."""
    model = SentenceTransformer("bert-base-nli-mean-tokens")
    for col in columns:
        # Create embeddings and add as a new column, skipping NA values
        df[f"{col}_embeddings"] = df[col].apply(
            lambda x: model.encode(x) if pd.notna(x) else None
        )
    return df


def search_semantic(
    query: str, df: pd.DataFrame, valid_embeddings_df: pd.DataFrame, top_k: int
) -> pd.DataFrame:
    """Search for similar entries using semantic search.

    Args:
        query (str): _description_
        df (pd.DataFrame): _description_
        valid_embeddings_df (pd.DataFrame): _description_
        top_k (int): _description_

    Returns:
        pd.DataFrame: _description_
    """
    if len(query) < 3:
        return df

    # empty df from previous search
    if len(df) < 1:
        return df

    model = SentenceTransformer("bert-base-nli-mean-tokens")
    logger.info("start embedding query")
    query_embedding = model.encode(query)
    logger.info("query embedded")

    logger.info(f"first 10 elements of query_embeddings are {query_embedding[:10]}")
    logger.info(f"there are {len(list(valid_embeddings_df))} valid embeddings")

    similarities = cosine_similarity(
        [query_embedding], list(valid_embeddings_df)
    ).flatten()

    logger.info(f"Get top_k:{top_k} similar entries using original indices")
    top_indices = similarities.argsort()[-top_k:][::-1]
    top_similar_indices = valid_embeddings_df.iloc[
        top_indices
    ].index  # Map to original DataFrame's indices
    top_similar = df.loc[top_similar_indices,]

    # drop the embedding columns when displaying on webapp
    top_similar = top_similar.loc[:, ~top_similar.columns.str.endswith("_embeddings")]

    return top_similar


@st.cache_resource
def _string_to_array(s: str) -> np.array:
    """Convert a string to a numpy array."""
    try:
        # Remove brackets and split by space
        values = s.strip("[]").split()
        # Convert to float and create array
        return np.array([float(x) for x in values])
    except ValueError:
        # Handle the exception in case of conversion error
        return np.array([])  # or some other fallback


@st.cache_resource
def return_colembed_array(
    df_indexed: pd.DataFrame, column_name: Union[str, List[str]]
) -> pd.DataFrame:
    """Return a DataFrame of embeddings from a column or columns of the dataframe."""

    def process_column(col_name: str, df: pd.DataFrame) -> pd.DataFrame:
        embedding_col = f"{col_name}_embeddings"
        logger.info(f"Check if the embedding column: {embedding_col} exists")

        if embedding_col not in df.columns:
            logger.error(f"Only available columns are '{df.columns}'")
            raise ValueError(
                f"Embedding column '{embedding_col}' does not exist in DataFrame."
            )

        return df[embedding_col].apply(_string_to_array)

    # Process based on the type of column_name
    if isinstance(column_name, str):
        return process_column(column_name, df_indexed)
    elif isinstance(column_name, list):
        # Initialize an empty DataFrame to hold all embeddings
        all_embeddings_df = pd.DataFrame()
        for col in column_name:
            col_embeddings_df = process_column(col, df_indexed)
            all_embeddings_df = pd.concat(
                [all_embeddings_df, col_embeddings_df], axis=1
            )
        return all_embeddings_df
    else:
        raise TypeError("column_name must be either a string or a list of strings.")


@st.cache_data
def load_indexed_data(path: str) -> pd.DataFrame:
    """Load indexed data."""
    EMBEDDING_SIZE = 768
    logger.info(f"Loading indexed file from {path}")
    df_indexed = pd.read_csv(path, compression="gzip", header=0)
    embedding_cols = [col for col in df_indexed.columns if col.endswith("_embeddings")]
    df_indexed[embedding_cols] = df_indexed[embedding_cols].fillna(
        str(np.zeros(EMBEDDING_SIZE))
    )
    return df_indexed


@st.cache_resource
def load_csv_data(file_path: str) -> pd.DataFrame:
    """Read the CSV file."""
    df = pd.read_csv(file_path, compression="gzip")
    return df
