import os

#OPENAI_API_KEY = ""
#TOGETHER_API_KEY = ""
#LLAMA_CLOUD_API_KEY = ""
#COHERE_API_KEY = ""

EVAL_NAME = "ACME_SPD_MINI" 
EVAL_DIRECTORY = "./datasets/acme_spd/files" 
EVAL_FILE = "./datasets/acme_spd/files/ACME_SPD.pdf"
EVAL_QUESTIONS = "./datasets/acme_spd/questions/ACME_SPD_Questions_Mini.json"
EVAL_RESULTS_DIR = "./datasets/acme_spd/evaluations/llamaindex/data"
EVAL_QUICK_TEST = "Are bifocals covered?"

GENERATION_LLM_FAMILY = "COHERE" # OPENAI, COHERE
GENERATION_LLM_MODEL = "command-r" # gpt-4, command-r

EMBEDDING_LLM_FAMILY = "COHERE" # OPENAI, COHERE
EMBEDDING_LLM_MODEL = "embed-english-v3.0" # text-embedding-3-large, embed-english-v3.0
EMBEDDING_DIMESIONS = 1024

EVALUATION_LLM_FAMILY = "OPENAI" # OPENAI, COHERE
EVALUATION_LLM_MODEL = "gpt-4-0125-preview" # gpt-4-0125-preview, command-r

RAG_STRATEGY = "S003_00"

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


EVAL_RESULTS_FILE = "F:/rag_sdk/datasets/acme_spd/evaluations/data/ACME_SPD_MINI_2024-05-05.xlsx"

def set_environment():
    variable_dict = globals().items()
    for key, value in variable_dict:
        #if "API" in key or "ID" in key:
        os.environ[key] = str(value)



