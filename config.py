import os

#OPENAI_API_KEY = ""
#TOGETHER_API_KEY = ""
#LLAMA_CLOUD_API_KEY = ""
#COHERE_API_KEY = ""
#ANTHROPIC_API_KEY = ""
#GOOGLE_API_KEY = ""
#LANGCHAIN_API_KEY = ""


EVAL_NAME = "SOW_CBA_01" 
EVAL_DIRECTORY = "F:/rag_sdk/datasets/sow_cba/files/md" 
EVAL_FILE = "F:/rag_sdk/datasets/sow_cba/files/md/"
EVAL_QUESTIONS = "F:/rag_sdk/datasets/sow_cba/questions/SOW_CBA_01_QA.xlsx"
EVAL_RESULTS_DIR = "F:/rag_sdk/evaluations/data/sow_cba"
EVAL_QUICK_TEST = "Who is my employer?"
#EVAL_DB = "F:/rag_sdk/datasets/sow_cba/db/SOW_CBA_01_OAI"
EVAL_DB = "F:/rag_sdk/datasets/sow_cba/db/SOW_CBA_01_COH"
EVAL_PROMPTS_DIR = "F:/rag_sdk/datasets/sow_cba/prompts" 

# OPENAI, COHERE, ANTHROPIC, GOOGLE, META, QWEN, MISTRALAI
GENERATION_LLM_FAMILY = "COHERE" 

# gpt-4o, command-r, command-r-plus, claude-3-5-sonnet-20240620, models/gemini-1.5-pro, 
# meta-llama/Llama-3-70b-chat-hf, meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo, meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo, 
# Qwen/Qwen2-72B-Instruct, mistralai/Mixtral-8x22B-Instruct-v0.1
#accounts/fireworks/models/llama-v3p1-405b-instruct, accounts/fireworks/models/llama-v3p1-70b-instruct

GENERATION_LLM_MODEL = "command-r"

# OPENAI, COHERE, GOOGLE
EMBEDDING_LLM_FAMILY = "COHERE" 

# text-embedding-3-large, embed-english-v3.0, models/text-embedding-004
EMBEDDING_LLM_MODEL = "embed-english-v3.0" 
EMBEDDING_DIMESIONS = 1024

EVALUATION_LLM_FAMILY = "OPENAI" # OPENAI, COHERE
EVALUATION_LLM_MODEL = "gpt-4-0125-preview" # gpt-4o, gpt-4-0125-preview

RAG_STRATEGY = "S007_00"

# Common Setting
CHUNK_SIZE = 512
SIMILARITY_TOP_K = 30
SIMILARITY_CUTOFF = 0.2 

#Reranker settings
RERANKER = "COHERE" #Will add others later
RERANKER_MODEL = "rerank-english-v3.0"
RERANK_TOP_N = 10

# S002 -> Sentence Window Retriever Settings
WINDOW_SIZE = 5

# S003 -> Recursive Retriever Settings
PARENT_CHUNK_SIZE = 1024
SUB_CHUNK_SIZES = "128_256_512"

# S004 -> Fusion Retriever Settings
RETRIEVER_WEIGHTS = "0.5_0.5"
FUSION_RERANKER = "reciprocal_rerank" # reciprocal_rerank, relative_score, dist_based_score, simple


#S008 -> Black Box 
BB_OUTPUT_FILE = "F:/rag_sdk/evaluations/black_box/ALB_NON_UNION_BEN_01_FAI_07_30.xlsx"

#S009 -> Retriever 
RTR_OUTPUT_FILE = ""


# Standard Langchain prompt
rag_prompt_lc_01 = """You are an assistant for question-answering tasks. 
Use the retrieved context, consisting of these documents, to answer the question. 
If you don't know the answer, just say that you don't know. 
Provide a detailed response, but do not invent stuff
\nContext: {context}
\nQuestion: {question}
"""

# Standard Cohere preamble
rag_prompt_coh_01 = """## Task & Context
You are an expert Human Resources assistant that helps employees answer questions about company policies. \
Use the provided documents to answer questions about an employee's specific situation.

## Style Guide
- Think step by step, provide evidence and/or reasoning first, then the answer."""

RAG_PROMPT_TEMPLATE = rag_prompt_lc_01

def set_environment():
    variable_dict = globals().items()
    for key, value in variable_dict:
        #if "API" in key or "ID" in key:
        os.environ[key] = str(value)



