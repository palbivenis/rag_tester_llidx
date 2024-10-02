import os
import re
import random
from datetime import datetime
import json
import asyncio
import pandas as pd
import argparse

from langsmith import Client
from langchain.callbacks.tracers import LangChainTracer
from langchain_core.tracers.context import tracing_v2_enabled
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain_text_splitters import MarkdownHeaderTextSplitter

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma

from langchain_anthropic import ChatAnthropic
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_together import ChatTogether
from langchain_fireworks import ChatFireworks

from llama_index.core import Settings
from llama_index.core.base.response.schema import Response
from llama_index.core.evaluation import BatchEvalRunner, CorrectnessEvaluator
from llama_index.llms.openai import OpenAI

from config_keys import set_keys
from evaluation_utils import threadpool_map

import logging


def choose_generation_llm(generation_llm_family, generation_llm_model):
    """Choose the LLM for generation based on parameters."""
    if generation_llm_family == "OPENAI":
        return ChatOpenAI(model_name=generation_llm_model, temperature=0)
    elif generation_llm_family == "ANTHROPIC":
        return ChatAnthropic(model_name=generation_llm_model, temperature=0)
    elif generation_llm_family == "GOOGLE":
        return ChatGoogleGenerativeAI(model=generation_llm_model, temperature=0)
    elif generation_llm_family == "COHERE":
        return ChatCohere(model=generation_llm_model, temperature=0)
    elif generation_llm_family == "META":
        return ChatFireworks(model=generation_llm_model, temperature=0)
    elif generation_llm_family in {"QWEN", "MISTRALAI"}:
        return ChatTogether(model=generation_llm_model, temperature=0)
    else:
        raise ValueError(f"Unsupported GENERATION_LLM_FAMILY: {generation_llm_family}")

def choose_embedding_model(embedding_llm_family, embedding_llm_model):
    """Choose the embedding model based on parameters."""
    if embedding_llm_family == "OPENAI":
        return OpenAIEmbeddings(model=embedding_llm_model)
    elif embedding_llm_family == "GOOGLE":
        return GoogleGenerativeAIEmbeddings(model=embedding_llm_model)
    elif embedding_llm_family == "COHERE":
        return CohereEmbeddings(model=embedding_llm_model)
    else:
        raise ValueError(f"Unsupported EMBEDDING_LLM_FAMILY: {embedding_llm_family}")

def populate_headers_to_split_on(i):
    """Return header tuples up to the specified level."""
    headers = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
        ("#####", "Header 5"),
    ]
    return headers[:i]

# Post-processing

def format_documents(retrieved_chunks):
    """Format retrieved documents for the prompt."""
    result = "\n"
    for chunk in retrieved_chunks:
        headers = [
            chunk.metadata.get("Header 1", ""),
            chunk.metadata.get("Header 2", ""),
            chunk.metadata.get("Header 3", ""),
            chunk.metadata.get("Header 4", ""),
            chunk.metadata.get("Header 5", ""),
        ]
        parents = [header for header in headers if header]
        title = parents[-1] if parents else "Untitled"
        parents_concat = '\n'.join(parents)
        result += (
            f"\n# Relevant Document Title:\n{title}\n"
            f"## Document Text:\n{chunk.page_content}\n"
            f"## This document is contained under the following sections:\n{parents_concat}\n"
        )
    return result

def initialize_vectorstore(eval_db, eval_directory, embeddings_model, header_levels, embedding_llm_model):
    """Initialize or create the vector store."""
    # Create a unique path for the vectorstore based on the embedding model
    embedding_model_name = embedding_llm_model.replace('/', '_')

    if os.path.exists(eval_db) and os.path.isdir(eval_db):
        logging.info(f"Loading existing vectorstore from {eval_db}")
        return Chroma(persist_directory=eval_db, embedding_function=embeddings_model)
    else:
        logging.info("Creating new vectorstore")

        text_loader_kwargs={'autodetect_encoding': True}
        loader = DirectoryLoader(eval_directory, glob="**/*.md", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)

        #loader = DirectoryLoader(eval_directory, glob="**/*.md", loader_cls=TextLoader)
        text_data = loader.load()
        page_contents = [item.page_content for item in text_data]
        text_concatenated = "\n\n".join(page_contents)

        headers_to_split_on = populate_headers_to_split_on(header_levels)
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=False)
        md_header_splits = markdown_splitter.split_text(text_concatenated)

        for chunk in md_header_splits:
            headers = [
                chunk.metadata.get("Header 1", ""),
                chunk.metadata.get("Header 2", ""),
                chunk.metadata.get("Header 3", ""),
                chunk.metadata.get("Header 4", ""),
                chunk.metadata.get("Header 5", ""),
            ]
            parents = [header for header in headers if header]
            parents_concat = f"{','.join(parents)}"
            title = parents[-1] if parents else "Untitled"
            pattern = re.compile(rf'#+\s+{re.escape(title)}')
            chunk.page_content = pattern.sub(parents_concat, chunk.page_content)

        vectorstore = Chroma.from_documents(
            documents=md_header_splits,
            embedding=embeddings_model,
            persist_directory=eval_db
        )
        vectorstore.persist()
        logging.info(f"Vectorstore created and persisted at {eval_db}")
        return vectorstore

def run_quick_test(rag_chain, eval_quick_test):
    """Run a quick test and print the response."""
    response = rag_chain.invoke(eval_quick_test)
    logging.info(f"Question:\n{eval_quick_test}\n")
    logging.info(f"Response:\n{response}\n")

def run_rag_pipeline(row, rag_chain, eval_name, batch_id, rag_strategy, rag_strategy_desc, similarity_top_k, generation_llm_model, embedding_llm_model, embedding_dimensions):
    """Run the RAG pipeline for a single query."""

    metadata = {
        "eval_name": eval_name,
        "batch_id": batch_id,
        "query_num": row["query_num"],
        "rag_strategy": rag_strategy,
        "rag_strategy_desc": rag_strategy_desc,
        "parameter_1": similarity_top_k,
        "parameter_2": "",
        "parameter_3": "",
        "parameter_4": "",
        "parameter_5": "",
        "model": generation_llm_model,
        "embed_model": embedding_llm_model,
        "embed_dimensions": embedding_dimensions,
    }

    logging.info(f"Processing query_num: {row['query_num']}")
    try:
        with tracing_v2_enabled(project_name=eval_name):
            response = rag_chain.invoke(row["query"], {"metadata": metadata})

        return {
            "query_num": row["query_num"],
            "generated_answer": response
        }
    except Exception as e:
        logging.exception(f"An error occurred while processing query_num: {row['query_num']}")
        return {
            "query_num": row["query_num"],
            "generated_answer": None
        }

async def main():
    """Main function to execute the evaluation pipeline."""
    set_keys()

    # Read the configuration file

    parser = argparse.ArgumentParser(description="Execute the evaluation pipeline with a specified configuration file.")
    parser.add_argument('config_file', type=str, help="Path to the configuration file")
    args = parser.parse_args()
    with open(args.config_file, 'r') as f:
        config = json.load(f)

    model_pairs = config['model_pairs']
    other_configurations = config['other_configurations']
    rag_prompt_template = config['rag_prompt_template']

    # Extract other configurations
    eval_name = other_configurations["EVAL_NAME"]
    eval_directory = other_configurations["EVAL_DIRECTORY"]
    eval_file = other_configurations["EVAL_FILE"]
    eval_questions = other_configurations["EVAL_QUESTIONS"]
    eval_results_dir = other_configurations["EVAL_RESULTS_DIR"]
    eval_quick_test = other_configurations["EVAL_QUICK_TEST"]
    rag_strategy = other_configurations["RAG_STRATEGY"]
    similarity_top_k = int(other_configurations["SIMILARITY_TOP_K"])
    header_levels = int(other_configurations["HEADER_LEVELS"])
    prompt_template = rag_prompt_template["template"]
    evaluation_llm_family = other_configurations["EVALUATION_LLM_FAMILY"]
    evaluation_llm_model = other_configurations["EVALUATION_LLM_MODEL"]
    embedding_dimensions = int(other_configurations["EMBEDDING_DIMENSIONS"])
    eval_log_dir = other_configurations["EVAL_LOG_DIR"]

    # Create log directory if it doesn't exist
    if not os.path.exists(eval_log_dir):
        os.makedirs(eval_log_dir)

    # Set up logging
    log_file = os.path.join(eval_log_dir, f"{eval_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s:%(message)s',
                        filename=log_file,
                        filemode='a')

    # Also add a console handler if desired
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    logging.info("Starting evaluation pipeline")

    # Load the evaluation questions
    queries = pd.read_excel(eval_questions)

    # Loop over each model pair
    for pair in model_pairs:
        try:
            # Extract the LLM and embedding model information
            generation_llm_family = pair["generation_llm_family"]
            generation_llm_model = pair["generation_llm_model"]
            embedding_llm_family = pair["embedding_llm_family"]
            embedding_llm_model = pair["embedding_llm_model"]
            eval_db = pair["EVAL_DB"]

            # Log the models being used
            logging.info(f"Processing model pair: Generation LLM Family: {generation_llm_family}, Model: {generation_llm_model}; Embedding LLM Family: {embedding_llm_family}, Model: {embedding_llm_model}")

            # Set environment variables or pass parameters directly
            os.environ["GENERATION_LLM_FAMILY"] = generation_llm_family
            os.environ["GENERATION_LLM_MODEL"] = generation_llm_model
            os.environ["EMBEDDING_LLM_FAMILY"] = embedding_llm_family
            os.environ["EMBEDDING_LLM_MODEL"] = embedding_llm_model

            # Choose the LLM for generation
            logging.info("Choosing the LLM for generation")
            llm = choose_generation_llm(generation_llm_family, generation_llm_model)

            # Choose the LLM for embedding
            logging.info("Choosing the LLM for embedding")
            embeddings_model = choose_embedding_model(embedding_llm_family, embedding_llm_model)

            # Prepare batch_id and other identifiers
            embed_string = embedding_llm_model.replace("models/", "") if "models/" in embedding_llm_model else embedding_llm_model
            generation_string = generation_llm_model.replace("meta-llama/", "").replace("accounts/fireworks/models/", "").replace("Qwen/", "").replace("models/", "").replace("mistralai/", "")

            if rag_strategy == "S007_00":
                rag_strategy_desc = "Semantic"
                batch_id = f"{eval_name}_{rag_strategy}_GM_{generation_string}_EM_{embed_string}_K_{similarity_top_k}_{random.randint(0, 999):03}"
            else:
                # Handle other RAG strategies if necessary
                raise ValueError(f"Unsupported RAG_STRATEGY: {rag_strategy}")

            logging.info(f"Batch ID: {batch_id}")

            output_file = os.path.join(eval_results_dir, f"{batch_id}.xlsx")

            # Setup Langsmith tracing
            os.environ['LANGCHAIN_TRACING_V2'] = 'true'
            os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
            os.environ['LANGCHAIN_PROJECT'] = eval_name

            logging.info("Setting up Langsmith tracing")

            # Initialize or create the vector store
            logging.info("Initializing or creating the vector store")
            vectorstore = initialize_vectorstore(eval_db, eval_directory, embeddings_model, header_levels, embedding_llm_model)
            retriever = vectorstore.as_retriever(search_kwargs={"k": similarity_top_k})

            # Prepare the prompt
            logging.info("Preparing the prompt")
            prompt = ChatPromptTemplate.from_template(prompt_template)

            # Prepare the RAG chain
            logging.info("Preparing the RAG chain")
            rag_chain = (
                {"context": retriever | format_documents, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            # Quick Test
            if eval_quick_test:
                logging.info("Running quick test")
                run_quick_test(rag_chain, eval_quick_test)

            # Prepare data for threading
            logging.info("Preparing data for threading")
            query_rows = [{"row": item[1]} for item in queries.iterrows()]

            args_list = []
            for item in query_rows:
                args = {
                    'row': item['row'],
                    'rag_chain': rag_chain,
                    'eval_name': eval_name,
                    'batch_id': batch_id,
                    'rag_strategy': rag_strategy,
                    'rag_strategy_desc': rag_strategy_desc,
                    'similarity_top_k': similarity_top_k,
                    'generation_llm_model': generation_llm_model,
                    'embedding_llm_model': embedding_llm_model,
                    'embedding_dimensions': embedding_dimensions,
                }
                args_list.append(args)

            # Run the RAG pipeline in parallel
            logging.info("Running the RAG pipeline in parallel")
            results = threadpool_map(
                run_rag_pipeline,
                args_list,
                num_workers=2,
                return_exceptions=True
            )

            # Merge results with queries
            logging.info("Merging results with queries")
            df = queries.merge(pd.DataFrame(results), on="query_num", how="inner")
            assert len(df) == len(queries), "Not all queries have been processed."

            # Choose the LLM for evaluations
            logging.info("Choosing the LLM for evaluations")
            if evaluation_llm_family == "OPENAI":
                Settings.eval_llm = OpenAI(temperature=0, model=evaluation_llm_model)
            else:
                raise ValueError(f"Unsupported EVALUATION_LLM_FAMILY: {evaluation_llm_family}")

            # Run evaluation
            logging.info("Running evaluation")
            eval_lidx_c = CorrectnessEvaluator(llm=Settings.eval_llm)

            runner = BatchEvalRunner(
                {"correctness": eval_lidx_c},
                workers=16,
            )

            LI_eval_results = await runner.aevaluate_responses(
                queries=df["query"].tolist(),
                responses=[Response(response=x) for x in df["generated_answer"].tolist()],
                reference=[{"reference": x} for x in df["expected_answer"].tolist()],
            )

            df["correctness_result"] = LI_eval_results["correctness"]
            df["correctness_llm"] = df["correctness_result"].map(lambda x: x.score)
            df["feedback_llm"] = df["correctness_result"].map(lambda x: x.feedback)
            logging.info(f"Average correctness score: {df['correctness_llm'].mean()}")

            # Prepare responses DataFrame
            responses_df = df[['query_num', 'query', 'expected_answer', 'generated_answer', 'correctness_llm']].copy()
            responses_df['correctness_human'] = responses_df['correctness_llm']
            responses_df[['faithfulness_llm', 'faithfulness_human']] = ""
            responses_df['rag_strategy'] = rag_strategy
            responses_df['rag_strategy_desc'] = rag_strategy_desc
            responses_df['parameter_1'] = similarity_top_k
            responses_df[['parameter_2', 'parameter_3', 'parameter_4', 'parameter_5']] = ""
            responses_df['model'] = generation_string
            responses_df['embed_model'] = embedding_llm_model
            responses_df['eval_model'] = evaluation_llm_model
            responses_df['embed_dimensions'] = embedding_dimensions
            responses_df['reranker'] = ""
            responses_df['run_date'] = datetime.today().strftime('%Y-%m-%d')
            responses_df['eval_name'] = eval_name
            responses_df['batch_id'] = batch_id

            # Get Performance Metrics from Langsmith
            logging.info("Waiting for performance metrics to become available")
            await asyncio.sleep(120)  # Wait for metrics to become available

            client = Client()
            runs = client.list_runs(
                project_name=eval_name,
                filter=f"and(eq(metadata_key, 'batch_id'), eq(metadata_value, '{batch_id}'))",
                is_root=True
            )

            usage_data = []
            for run in runs:
                metadata = run.extra.get("metadata", {})
                usage_data.append({
                    "query_num": metadata.get("query_num"),
                    "total_tokens": run.total_tokens,
                    "prompt_tokens": run.prompt_tokens,
                    "completion_tokens": run.completion_tokens,
                    "total_cost": f"${run.total_cost:.4f}" if run.total_cost else None,
                    "prompt_cost": f"${run.prompt_cost:.4f}" if run.prompt_cost else None,
                    "completion_cost": f"${run.completion_cost:.4f}" if run.completion_cost else None,
                    "latency": (run.end_time - run.start_time).total_seconds() if run.end_time else None,
                    "first_token_ms": (run.first_token_time - run.start_time).total_seconds() * 1000 if run.first_token_time else None,
                })

            usage_df = pd.DataFrame(usage_data)
            responses_df = responses_df.merge(usage_df, on='query_num', how='left')

            # Create summary DataFrame
            correctness_sum = df['correctness_llm'].sum()
            correctness_mean = df['correctness_llm'].mean()
            summary_df = pd.DataFrame({
                'Metric': ['Sum', 'Mean'],
                'Value': [correctness_sum, correctness_mean]
            })

            # Prepare correctness DataFrame
            correctness_df = df[['query_num', 'query', 'expected_answer', 'generated_answer', 'correctness_llm', 'feedback_llm']].copy()
            correctness_df['correctness_human'] = correctness_df['correctness_llm']
            correctness_df['feedback_human'] = ""
            correctness_df['batch_id'] = batch_id

            # Write all DataFrames to Excel
            logging.info(f"Saving results to {output_file}")
            with pd.ExcelWriter(output_file) as writer:
                responses_df.to_excel(writer, sheet_name="Responses", index=False)
                summary_df.to_excel(writer, sheet_name="Summary", index=False)
                correctness_df.to_excel(writer, sheet_name="Correctness", index=False)

            logging.info(f"Results have been saved to {output_file}")
            logging.info(f"Finished processing model pair: {pair}")

        except Exception as e:
            logging.exception(f"An error occurred while processing model pair {pair}. Skipping to next model pair.")
            continue

if __name__ == "__main__":
    asyncio.run(main())
