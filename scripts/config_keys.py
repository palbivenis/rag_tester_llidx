import os

#OPENAI_API_KEY = ""
#TOGETHER_API_KEY = ""
#LLAMA_CLOUD_API_KEY = ""
#COHERE_API_KEY = ""
#ANTHROPIC_API_KEY = ""
#GOOGLE_API_KEY = ""
#LANGCHAIN_API_KEY = ""



def set_keys():
    variable_dict = globals().items()
    for key, value in variable_dict:
        #if "API" in key or "ID" in key:
        os.environ[key] = str(value)



