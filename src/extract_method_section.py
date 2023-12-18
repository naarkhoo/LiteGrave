"""Extract material and method section to extract device, cell type and time."""

import pickle
from glob import glob

from dotenv import load_dotenv
from loguru import logger
from omegaconf import OmegaConf
from tqdm import tqdm

from utils_llm import count_tokens, format_dataframe_to_string

cfg = OmegaConf.load("conf/config.yaml")

# Load environment variables from .env file
load_dotenv(dotenv_path=cfg.credentials.path)

pkl_files = glob(f"{cfg.data.pdf_preprocessed_path}/*.pkl")
logger.info(f"Found {len(pkl_files)} pkl files")

for pkl_file in tqdm(pkl_files):
    logger.info(f"Start processing {pkl_file}")
    # pkl_file = f"{cfg.data.pdf_preprocessed_path}/27117806.pkl"
    with open(pkl_file, "rb") as f:
        paper_summary_obj = pickle.load(f)

    paper_summary_df = paper_summary_obj["dataframe"]
    logger.info(f"Number of chunks in {pkl_file}: {len(paper_summary_df)}")

    paper_summary_df.page = [int(page_id) for page_id in paper_summary_df.page]

    # first the first page
    min_page_id = min(paper_summary_df["page"])
    logger.info(f"the first page is: {min_page_id}")

    paper_summary_df = paper_summary_df.query(f"page <= {min_page_id + 1}").copy()
    paper_summary_df["section"] = [item.lower() for item in paper_summary_df["section"]]
    logger.info(f"Number of chunks in first two pages: {len(paper_summary_df)}")
    logger.info(
        f"Section title for the first two pages: {paper_summary_df['section'].unique()}"
    )

    # don't put discussion - 32900939 first two pages are result and discussion
    exclude_list = ["ethic", "declaration of interes", "statistical"]

    # Filtering the DataFrame
    for keyword in exclude_list:
        paper_summary_df = paper_summary_df[
            ~paper_summary_df["section"].str.contains(keyword, case=False, na=False)
        ]
    logger.info(
        f"""Number of chunks in first two pages excluding {exclude_list}:
        {len(paper_summary_df)}"""
    )

    # Step 1: Create an ordered list of unique sections
    ordered_sections = paper_summary_df["section"].drop_duplicates().tolist()

    # Step 2: Group by 'section' and concatenate 'text'
    grouped_df = paper_summary_df.groupby("section")["text"].agg(" ".join).reset_index()

    # Reordering the grouped DataFrame based on the order of sections
    grouped_df = grouped_df.set_index("section").reindex(ordered_sections).reset_index()

    formatted_data = format_dataframe_to_string(grouped_df)
    logger.info(f"Number of tokens in context: {count_tokens(formatted_data)}")

    context_length = count_tokens(formatted_data)

    # save both formatted_data and context_length to a file
    data_to_save = {
        "context_length": context_length,
        "context": formatted_data,
    }
    # save data_to_save into a pickle file
    with open(pkl_file.replace(".pkl", "_f2page.pkl"), "wb") as f:
        pickle.dump(data_to_save, f)

logger.info("Done")
