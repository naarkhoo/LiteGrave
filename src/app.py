"""Streamlit application for Search."""

import base64
import os
from pathlib import Path
from typing import Optional

import streamlit as st
from langchain.agents.agent_types import AgentType
from langchain.llms import OpenAI
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from loguru import logger
from omegaconf import OmegaConf

from utils import (
    exact_match,
    load_csv_data,
    load_indexed_data,
    return_colembed_array,
    search_semantic,
)

# Load the configuration
cfg = OmegaConf.load("conf/config.yaml")

LOGO_IMAGE = "src/logo.png"
LOGO_WIDTH = 200
CSV_FILE_PATH = "data/m44.csv.gz"
SHOW_CSV = True
SINGLE_SEARCH_BAR = False
LAYOUT2X2 = True
LAYOUT1X4 = False

st.set_page_config(page_title="Gene and Cell Study Predictor", layout="wide")

df_indexed = load_indexed_data(cfg.data.indexed_path)
duration_embedding_df = return_colembed_array(
    df_indexed, column_name=["experiment time", "device"]
)

# create langchain agent
agent = create_csv_agent(
    OpenAI(temperature=0),
    cfg.data.path,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)


@st.cache_resource
def load_css(css_path: str) -> None:
    """Load a CSS file."""
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_resource
def img_to_bytes(img_path: str) -> str:
    """Convert an image file to bytes."""
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


def img_to_html(img_path: str, width: Optional[int] = None) -> str:
    """Convert an image file to HTML elements.

    Args:
        img_path (str): _description_
        width (Optional[int], optional): _description_. Defaults to None.

    Returns:
        str:
    """
    img_data = (
        f"<img src='data:image/png;base64,{img_to_bytes(img_path)}' class='img-fluid'"
    )

    # If width is specified
    if width:
        img_data += f" width='{width}'"

    img_data += ">"
    return img_data


def main() -> None:
    """Main function of the App."""
    # Set the current working directory (for debugging purposes).
    print(f"Current working directory: {os.getcwd()}")

    # Load the external CSS.
    load_css("src/style.css")

    logo_html = img_to_html(LOGO_IMAGE, width=LOGO_WIDTH)
    st.markdown(f'<div class="centered">{logo_html}</div>', unsafe_allow_html=True)

    st.markdown("".join(["<br>"] * 2), unsafe_allow_html=True)

    # Centered search bar in the main area.
    if SINGLE_SEARCH_BAR:
        st.markdown('<div class="search-container">', unsafe_allow_html=True)
        search_query = st.text_input("Search", key="search")
        st.markdown("</div>", unsafe_allow_html=True)

        # If a search query is entered
        if search_query:
            # add a placeholder
            # results = search_google(search_query)
            results = [("Google", "https://google.com")]

            # Display the search results.
            st.write("Search results:")
            for result in results:
                st.markdown(
                    f'<a href="{result[1]}">{result[0]}</a>', unsafe_allow_html=True
                )

    if LAYOUT1X4:
        # Creating columns for each input field
        col_cell_type, col_environment, col_duration, col_target = st.columns(4)

        with col_cell_type:
            cell_type_query = st.text_area(
                "Cell Type", placeholder="Enter the cell type (e.g., HeLa cells)"
            )

        with col_environment:
            environment_query = st.text_area(
                "Environment",
                placeholder="Describe the environment condition (e.g., hypoxic)",
            )

        with col_duration:
            duration_query = st.text_area(
                "Duration", placeholder="Experiment duration (e.g., 3 days)"
            )

        with col_target:
            target = st.text_area(
                "Gene/Pathway Target",
                placeholder="Specify gene or pathway (e.g., P53 signaling)",
            )

    if LAYOUT2X2:
        col1, col2 = st.columns(2)

        with col1:
            cell_type_query = st.text_area(
                "Cell Type", placeholder="Enter the cell type (e.g., HeLa cells)"
            )
            environment_query = st.text_area(
                "Environment",
                placeholder="Describe the environment condition (e.g., hypoxic)",
            )

        with col2:
            duration_query = st.text_area(
                "Duration", placeholder="Experiment duration (e.g., 3 days)"
            )
            target = st.text_area(
                "Gene/Pathway Target",
                placeholder="Specify gene or pathway (e.g., P53 signaling)",
            )

    st.markdown("".join(["<br>"] * 12), unsafe_allow_html=True)

    left_col, col5, col6, right_col = st.columns([5, 1, 1, 5])
    with col5:
        explore_studies_btn = st.button("Explore Studies")

    with col6:
        run_prediction_btn = st.button("Run Prediction")

    if explore_studies_btn:
        # Logic to show existing studies
        st.write(
            "Showing studies for:",
            cell_type_query,
            environment_query,
            duration_query,
            target,
        )

    if run_prediction_btn:
        # Logic to predict the outcome
        st.write(
            "Predicting outcome for:",
            cell_type_query,
            environment_query,
            duration_query,
            target,
        )

    # Ensure that all inputs are provided before any action is taken.
    if (explore_studies_btn or run_prediction_btn) and not (
        cell_type_query and environment_query and duration_query and target
    ):
        st.error("Please fill in all fields to proceed.")

    if SHOW_CSV:
        # Call the function to load the data
        data = load_csv_data(CSV_FILE_PATH)

        # Convert the PubMed ID to a string so you avoid the comma separator
        data = data.astype({"pubmed_id": str})
        original_data = data.copy()

        # Filter the data
        data_cell_type = exact_match(
            df=original_data, colname="cell type", term=cell_type_query
        )
        logger.info(f"exact match resulted in {len(data_cell_type)} rows")

        data_duration = search_semantic(
            duration_query,
            original_data,
            duration_embedding_df["experiment time_embeddings"],
            top_k=3,
        )

        data_device = search_semantic(
            environment_query,
            original_data,
            duration_embedding_df["device_embeddings"],
            top_k=3,
        )

        common_pubmed_ids = (
            set(data_cell_type["pubmed_id"])
            .intersection(set(data_duration["pubmed_id"]))
            .intersection(set(data_device["pubmed_id"]))
        )

        # Step 2: Subset the original data based on these common 'pubmed_id' values
        subset_data = original_data[original_data["pubmed_id"].isin(common_pubmed_ids)]

        # Use Streamlit to write the DataFrame to the app
        st.write("Displaying CSV data:")
        st.dataframe(subset_data)


if __name__ == "__main__":
    main()
