"""Streamlit application for Search."""

import base64
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

LOGO_IMAGE = "src/logo.png"
LOGO_WIDTH = 200
CSV_FILE_PATH = "data/m44.csv.gz"
SHOW_CSV = True
SINGLE_SEARCH_BAR = False
LAYOUT2X2 = True
LAYOUT1X4 = False

st.set_page_config(page_title="Gene and Cell Study Predictor", layout="wide")


@st.cache_resource
def load_css(css_path: str) -> None:
    """Load a CSS file."""
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


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


# Cache the loading of the CSV file to avoid unnecessary reloads
@st.cache_resource
def load_csv_data(file_path: str) -> pd.DataFrame:
    """Read the CSV file."""
    df = pd.read_csv(file_path, compression="gzip")
    return df


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
            cell_type = st.text_area(
                "Cell Type", placeholder="Enter the cell type (e.g., HeLa cells)"
            )

        with col_environment:
            environment = st.text_area(
                "Environment",
                placeholder="Describe the environment condition (e.g., hypoxic)",
            )

        with col_duration:
            duration = st.text_area(
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
            cell_type = st.text_area(
                "Cell Type", placeholder="Enter the cell type (e.g., HeLa cells)"
            )
            environment = st.text_area(
                "Environment",
                placeholder="Describe the environment condition (e.g., hypoxic)",
            )

        with col2:
            duration = st.text_area(
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
        st.write("Showing studies for:", cell_type, environment, duration, target)

    if run_prediction_btn:
        # Logic to predict the outcome
        st.write("Predicting outcome for:", cell_type, environment, duration, target)

    # Ensure that all inputs are provided before any action is taken.
    if (explore_studies_btn or run_prediction_btn) and not (
        cell_type and environment and duration and target
    ):
        st.error("Please fill in all fields to proceed.")

    if SHOW_CSV:
        # Call the function to load the data
        data = load_csv_data(CSV_FILE_PATH)

        # Convert the PubMed ID to a string so you avoid the comma separator
        data = data.astype({"pubmed_id": str})

        # Use Streamlit to write the DataFrame to the app
        st.write("Displaying CSV data:")
        st.dataframe(data)


if __name__ == "__main__":
    main()
