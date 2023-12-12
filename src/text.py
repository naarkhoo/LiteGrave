"""Test utils_llm.index_pdf() function."""
import os

from omegaconf import OmegaConf

from utils_llm import index_pdf

cfg = OmegaConf.load("conf/config.yaml")
input = os.path.join(cfg.data.pdf_path, "20213684.pdf")
output = index_pdf(input, cfg, write=True)


data_section = output["data_section"]
data_page = output["data_page"]
df = output["dataframe"]

print(df.head(4))
print(data_page.keys())
print(data_section.keys())
