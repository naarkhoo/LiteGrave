"""Extract device from a query using the few-shot learning approach."""
import pandas as pd
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from utils_llm import count_tokens, filter_by_cumulative_length

cfg = OmegaConf.load("conf/config.yaml")

# Load environment variables from .env file
load_dotenv(dotenv_path=cfg.credentials.path)

new_query = "Microgravity on space flights has been shown to affect the physiology of a cell considerably [1]. Normal gravity (1 g) affects 2-Dimensional culture by depositing cells on the surface of the tissue culture plate (TCP) where anchorage-dependent cells adhere and proliferate as a monolayer with very limited cell-cell interactions. The weightlessness and reduced acceleration (less than 1 g) in space, removes the effect of gravity, allowing cell cultures in space to have unhindered movement of the culture medium, a shear free environment and, as cells are not bound by any directional force, unrestricted movement of cells within the medium. Under such conditions cells tend to coalesce and form aggregates creating three dimensional (3D) environments where they interact on multiple planes [2]. The effect of reduced gravity is not restricted to changes in culture conditions as the unique environment can produce changes in the fundamental physiology of the cell. While the mechanism of action of how gravity, or the lack of it, affects molecular and cellular functions is still unclear, it has been established that microgravity or zero gravity affects vital processes of the cell and importantly, microgravity has been shown to alter cancer growth and progression [3][4][5]. However, different cancers respond differently to microgravity by losing or enhancing cellular processes and functions. In this study we cultured cell lines representative of solid and hematological tumors-DLD-1, MOLT-4 and HL-60 in a rotating cell culture system (RCCS) that simulated microgravity. The RCCS is a mechanical system that simulates reduced gravity on earth by canceling the directional vector through constant rotation of a High Aspect Ratio Vessel (HARV). This maintains cells in a constant free fall and a shear free environment allowing cells to coalesce and form 3D aggregates [2]. These aggregates are maintained in free fall and experience conditions of reduced gravity for the remainder of the culture period. We hypothesized that physiological changes to the cell functions such as cell proliferation and viability could be corroborated with changes in fundamental processes of the cell such as gene expression. To relate physiological changes such as an altered cell cycle profile with dysregulation of gene expression, real time PCR analysis for cell cycle genes, oncogenes and cancer development and progression markers was carried out. Genome wide expression profiling by DNA microarray of these cell lines cultured under microgravity revealed the dysregulation of several pathways in cancer and importantly, corroborated with observed physiological changes to the cell. We also used the gene expression profile to investigate dysregulation in pathways central to cancer such as the Notch signaling system and dysregulation in post transcriptional gene silencing machinery. The gene expression profile also revealed dysregulation of microRNA host genes in microgravity including the significant tumor suppressor, miR-22 in DLD-1."  # noqa: E501
logger.info(f"Number of tokens in the query: {count_tokens(new_query)}")

max_example_token_num = 4097 - count_tokens(new_query)  # 4097 is the max
logger.info(f"max number of token left for context is: {max_example_token_num}")


def return_labelled_data(cfg: DictConfig, type: str) -> pd.DataFrame:
    """Return the labelled data as a dataframe."""
    # read the data
    df = pd.read_csv(cfg.data.labeled_rag_path)
    logger.info(f"reading {len(df)} labeled data")
    if type not in list(df["key"].unique()):
        raise ValueError(f"{type} must be {set(df['key'].unique())}")

    # filter the dataframe to only include the device data
    df_device = df.dropna(subset=["context"]).query(f"key == '{type}' & value != 'Nan'")
    logger.info(f"found {len(df_device)} labeled data for {type}")

    return df_device


df_type = return_labelled_data(cfg, "device")

# iterate over the dataframe and create examples
examples = [
    {"input": input_value, "output": output_value}
    for input_value, output_value in zip(df_type["context"], df_type["value"])
]

# count the number of token per example
example_token_counts = [
    count_tokens(example["input"] + example["output"]) for example in examples
]
logger.info(f"char_counts: {example_token_counts}")

filtered_examples = filter_by_cumulative_length(
    examples, example_token_counts, max_example_token_num
)

# Define the example prompt template
example_prompt = ChatPromptTemplate.from_messages(
    [("human", "{input}"), ("ai", "{output}")]
)

# Create the few-shot prompt template
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=filtered_examples,
)

# Define the final prompt template
final_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI assistant helping to find what device has been used in a microgravity scientific literature; these devices are used",  # noqa: E501
        ),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)

chain = final_prompt | ChatOpenAI(temperature=0.0)
response = chain.invoke({"input": new_query})
print(response.content)
