import os

#OPENAI_API_KEY = ""
#TOGETHER_API_KEY = ""
#LLAMA_CLOUD_API_KEY = ""
#COHERE_API_KEY = ""

OPENAI_API_KEY = "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"
TOGETHER_API_KEY = "791b88ad42d74d2407df28271cd2f2e81602c51da02747ee5f397d07386264b1"
LLAMA_CLOUD_API_KEY = "llx-d6R1eNupbszqFBuQWdb0TvYPTEwX2oO9u3sduNYGfoGgCWPt"
COHERE_API_KEY = "3DGonhWxrH4Xwrfkh1L3TWfXSLSXp4S9N2ILWCpK"

EVAL_NAME = "ITPEU_SPD" 
EVAL_DIRECTORY = "./datasets/itpeu_spd/files" 
EVAL_FILE = "./datasets/itpeu_spd/files/ITPEU_SPD.pdf"
EVAL_QUESTIONS = "./datasets/itpeu_spd/questions/ITPEU_SPD_Questions.json"
EVAL_RESULTS_DIR = "./datasets/itpeu_spd/evaluations/llamaindex/data"
EVAL_QUICK_TEST = "My disabled daughter is 28 years old. Is she covered?"

GENERATION_LLM_FAMILY = "COHERE" # OPENAI, COHERE
GENERATION_LLM_MODEL = "command-r" # gpt-4, command-r

EMBEDDING_LLM_FAMILY = "COHERE" # OPENAI, COHERE
EMBEDDING_LLM_MODEL = "embed-english-v3.0" # text-embedding-3-large, embed-english-v3.0

EVALUATION_LLM_FAMILY = "OPENAI" # OPENAI, COHERE
EVALUATION_LLM_MODEL = "gpt-4-0125-preview" # gpt-4-0125-preview, command-r

RAG_STRATEGY = "S004_00"

# Common Setting
CHUNK_SIZE = 512
SIMILARITY_TOP_K = 3
SIMILARITY_CUTOFF = 0.2 

#Reranker settings
RERANKER = "COHERE" #Will add others later
RERANK_TOP_N = 10

# S002 -> Sentence Window Retriever Settings
WINDOW_SIZE = 5

# S003 -> Recursive Retriever Settings
PARENT_CHUNK_SIZE = 1024
SUB_CHUNK_SIZES = "128_256_512"

# S004 -> Fusion Retriever Settings
RETRIEVER_WEIGHTS = "0.5_0.5"
FUSION_RERANKER = "reciprocal_rerank" # reciprocal_rerank, relative_score, dist_based_score, simple


def set_environment():
    variable_dict = globals().items()
    for key, value in variable_dict:
        #if "API" in key or "ID" in key:
        os.environ[key] = str(value)



