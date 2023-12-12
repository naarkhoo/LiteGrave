"""Generate CSV file that summarizes the papers."""
from collections import Counter
from glob import glob

import inflect

from utils_llm import clean_section_names, read_jsonl_to_dict

all_section_files = glob("data/preprocessed_pdf/*_section.jsonl")

unique_keys = Counter()  # type: ignore[var-annotated]

p = inflect.engine()

for item in all_section_files:
    d = read_jsonl_to_dict(item)
    cleaned_keys = [clean_section_names(key, p) for key in d.keys()]
    unique_keys.update(cleaned_keys)

print(unique_keys)
