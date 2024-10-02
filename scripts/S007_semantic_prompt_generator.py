import os
import sys
import json
import logging
import asyncio
import pandas as pd

# Import embedding models

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_cohere import CohereEmbeddings

from config_keys import set_keys

# Define the async main function
async def main():
    # Check if config file is provided
    if len(sys.argv) != 2:
        print("Usage: python script.py config.json")
        sys.exit(1)

    # Load the config file
    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        config = json.load(f)

    set_keys()

    # Read global variables from config
    preamble = config.get('preamble', '')
    similarity_top_k = int(config.get('similarity_top_k', 5))
    log_dir = config.get('log_dir')

    # Create the log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Loop over each vector store
    for vector_store_config in config.get('vector_stores', []):
        logger = None  # Initialize logger
        try:
            # Read variables for this vector store
            vector_store_path = vector_store_config.get('vector_store_path')
            embedding_llm_family = vector_store_config.get('embedding_llm_family')
            embedding_llm_model = vector_store_config.get('embedding_llm_model')
            eval_name = vector_store_config.get('eval_name')
            eval_questions = vector_store_config.get('eval_questions')
            eval_quick_test = vector_store_config.get('eval_quick_test')
            eval_prompts_dir = vector_store_config.get('eval_prompts_dir')

            # Create the prompts directory if it doesn't exist
            os.makedirs(eval_prompts_dir, exist_ok=True)

            # Get the vector store name
            vector_store_name = os.path.basename(vector_store_path.rstrip('/\\'))

            # Set up logging for this vector store
            logger = logging.getLogger(vector_store_name)
            logger.setLevel(logging.INFO)

            # Create file handler
            log_file = os.path.join(log_dir, f"{vector_store_name}.log")
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.INFO)

            # Create console handler
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)

            # Create formatter and add it to the handlers
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)

            # Add handlers to the logger
            logger.addHandler(fh)
            logger.addHandler(ch)

            logger.info(f"Processing vector store: {vector_store_name}")

            # Set up the embeddings model
            if embedding_llm_family == "OPENAI":
                embeddings_model = OpenAIEmbeddings(model=embedding_llm_model)
            elif embedding_llm_family == "COHERE":
                embeddings_model = CohereEmbeddings(model=embedding_llm_model)
            else:
                raise ValueError(f"Unsupported embedding LLM family: {embedding_llm_family}")

            # Load the vectorstore
            vectorstore = Chroma(persist_directory=vector_store_path,
                                 embedding_function=embeddings_model)

            retriever = vectorstore.as_retriever(search_kwargs={"k": similarity_top_k})

            # Run the quick test
            retrieved_chunks = await asyncio.to_thread(retriever.get_relevant_documents, eval_quick_test)
            context = format_documents(retrieved_chunks)
            prompt = f"{preamble}{context}\nQuestion: {eval_quick_test}\n"

            logger.info(f"Quick test prompt:\n{prompt}")

            # Load the evaluation questions
            queries = pd.read_excel(eval_questions)

            # Process queries asynchronously
            tasks = [run_prompt_generator(row, retriever, preamble) for index, row in queries.iterrows()]
            results = await asyncio.gather(*tasks)

            df_results = pd.DataFrame(results)
            # df = queries.merge(df_results, on="query_num", how="inner")
            df = queries.merge(df_results, on=["query_num", "query"], how="inner")
            assert len(df) == len(queries), "Mismatch in query results"

            # Save results to files
            #output_file_xls = os.path.join(eval_prompts_dir, f"{eval_name}_{embedding_llm_family}_{embedding_llm_model}_P.xlsx")
            output_file_json = os.path.join(eval_prompts_dir, f"{eval_name}_{embedding_llm_family}_{embedding_llm_model}_P.json")

            #with pd.ExcelWriter(output_file_xls) as writer:
            #    df.to_excel(writer, sheet_name="Prompts", index=False)

            df.to_json(output_file_json, orient='records', lines=True)

            #logger.info(f"Results saved to {output_file_xls} and {output_file_json}")
            logger.info(f"Results saved to {output_file_json}")

        except Exception as e:
            if logger:
                logger.error(f"An error occurred while processing vector store {vector_store_name}: {e}", exc_info=True)
            else:
                # If logger is not initialized, print error to console
                print(f"An error occurred while processing vector store {vector_store_name}: {e}")
        finally:
            # Remove handlers after processing or error
            if logger:
                handlers = logger.handlers[:]
                for handler in handlers:
                    handler.close()
                    logger.removeHandler(handler)

# Define the function to format documents
def format_documents(retrieved_chunks):
    result = "\n"

    for chunk in retrieved_chunks:
        header_1 = chunk.metadata.get("Header 1", "")
        header_2 = chunk.metadata.get("Header 2", "")
        header_3 = chunk.metadata.get("Header 3", "")
        header_4 = chunk.metadata.get("Header 4", "")
        header_5 = chunk.metadata.get("Header 5", "")

        headers = [header_1, header_2, header_3, header_4, header_5]
        parents = []

        for header in headers:
            if header == "":
                break
            parents.append(header)

        # Identify the title as the last non-empty header
        title = parents[-1] if parents else "Untitled"

        parents_concat = '\n'.join(parents)

        result += (
            f"\n# Relevant Document Title:\n{title}\n"
            f"## Document Text:\n{chunk.page_content}\n"
            f"## This document is contained under the following sections:\n{parents_concat}\n"
        )

    return result

# Define the async function to generate prompts
async def run_prompt_generator(row, retriever, preamble):
    retrieved_chunks = await asyncio.to_thread(retriever.get_relevant_documents, row["query"])
    context = format_documents(retrieved_chunks)
    prompt = f"{preamble}{context}\nQuestion: {row['query']}\n"

    return {
        "query_num": row["query_num"],
        "query": row["query"],
        "prompt": prompt.replace('\\n', '\n'),
    }

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())

