import sys
import re
import json
import logging
import os
from pathlib import Path
import argparse
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tiktoken

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from openpyxl import Workbook
from openpyxl.drawing.image import Image

def load_config(config_path: Path) -> dict:
    """Load configuration parameters from a JSON file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logging.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file {config_path} not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {config_path}: {e}")
        sys.exit(1)

def setup_logging(log_directory: str):
    """Configure logging to console and a timestamped log file."""
    # Ensure the log directory exists
    log_dir = Path(log_directory)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Generate log file name based on current time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"process_log_{timestamp}.log"
    log_file_path = log_dir / log_filename

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Formatter for log messages
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    try:
        fh = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logging.info(f"Logging to file: {log_file_path}")
    except Exception as e:
        logging.error(f"Failed to set up file logging at {log_file_path}: {e}")

def populate_headers_to_split_on(levels: int) -> list:
    """Generate headers based on the specified header levels."""
    headers = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
        ("#####", "Header 5"),
    ]
    return headers[:levels]

def num_tokens_from_string(string: str, encoding) -> int:
    """Returns the number of tokens in a text string."""
    return len(encoding.encode(string))

def load_and_concatenate_text(input_dir: Path, loader_cls, loader_kwargs) -> str:
    """Load and concatenate text from all Markdown files in the input directory."""
    try:

        text_loader_kwargs={'autodetect_encoding': True}
        loader = DirectoryLoader(input_dir, glob="**/*.md", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
        text_data = loader.load()
        page_contents = [item.page_content for item in text_data]
        text_concatenated = "\n\n".join(page_contents)
        logging.info(f"Loaded and concatenated text from {input_dir}")
        return text_concatenated
    except Exception as e:
        logging.error(f"Error loading files from {input_dir}: {e}")
        return ""

def split_text(text: str, headers_to_split_on: list) -> list:
    """Split text into chunks based on Markdown headers."""
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=False)
    try:
        splits = splitter.split_text(text)
        logging.info(f"Text split into {len(splits)} chunks based on headers.")
        return splits
    except Exception as e:
        logging.error(f"Error splitting text: {e}")
        return []

def process_chunks(chunks, encoding, header_levels: int) -> dict:
    """Process each chunk to extract metadata and compute tokens."""
    documents = {}
    for i, chunk in enumerate(chunks):
        parents = [chunk.metadata.get(f"Header {level}", "") for level in range(1, header_levels + 1)]
        parents = [header for header in parents if header]
        parents_concat = ",".join(parents)
        
        title = parents[-1] if parents else "Untitled"
        pattern = re.compile(rf'#+\s+{re.escape(title)}')
        chunk.page_content = pattern.sub(parents_concat, chunk.page_content)
        
        tokens = num_tokens_from_string(chunk.page_content, encoding)
        document = {
            "doc_id": i + 1,
            "doc_length": len(chunk.page_content),
            "tokens": tokens,
            "headers": parents,
            "text": chunk.page_content,
            "title": title
        }
        documents[i + 1] = document
    logging.info(f"Processed {len(documents)} chunks into documents.")
    return documents

def write_chunked_markdown(documents: dict, output_md: Path):
    """Write processed documents to a chunked Markdown file."""
    try:
        with open(output_md, 'w', encoding='utf-8') as file:
            for doc_id, document in documents.items():
                file.write(f"## Document - {document['doc_id']}\n")
                file.write(f"**Tokens - {document['tokens']}**\n")
                file.write(f"**Text of this document:**\n\n{document['text']}\n")
                file.write(f"**Title of this document:**\n{document['title']}\n")
                file.write(f"**This document is contained under the following titles:**\n{','.join(document['headers'])}\n\n")
        logging.info(f"Wrote chunked Markdown to {output_md}")
    except Exception as e:
        logging.error(f"Error writing chunked Markdown to {output_md}: {e}")

def analyze_tokens(df: pd.DataFrame, token_thresholds: list) -> tuple:
    """Analyze token distributions and calculate percentiles."""
    percentiles = [50, 95, 99]
    token_count_percentiles = np.percentile(df["tokens"], percentiles)
    
    percentiles_df = pd.DataFrame({
        'Percentile': percentiles,
        'Token Count': token_count_percentiles
    })
    
    percentile_values = [(df["tokens"] <= p).mean() * 100 for p in token_thresholds]
    
    percentile_chunks_df = pd.DataFrame({
        'Token Count Threshold': token_thresholds,
        'Percentage of Chunks': percentile_values
    })
    
    logging.info("Token analysis completed.")
    return percentiles_df, percentile_chunks_df

def generate_excel_report(df: pd.DataFrame, percentiles_df: pd.DataFrame, 
                         percentile_chunks_df: pd.DataFrame, analysis_excel: Path, 
                         histogram_img: str, bins: int, token_thresholds: list):
    """Generate an Excel report with data and visualizations."""
    try:
        with pd.ExcelWriter(analysis_excel, engine='openpyxl') as writer:
            # Write Chunks sheet
            df.drop(columns=["text"]).to_excel(writer, sheet_name='Chunks', index=False)
            logging.info(f"Wrote 'Chunks' sheet to {analysis_excel}")
            
            # Generate and save histogram
            plt.figure(figsize=(10, 6))
            plt.hist(df["tokens"], bins=bins, edgecolor='black')
            plt.xlabel("Tokens")
            plt.ylabel("Frequency")
            plt.title("Histogram of Tokens")
            plt.savefig(histogram_img)
            plt.close()
            logging.info(f"Histogram saved as {histogram_img}")
            
            # Insert histogram into Excel
            wb = writer.book
            ws = wb.create_sheet('Histogram')
            img = Image(histogram_img)
            ws.add_image(img, 'A1')
            logging.info(f"Inserted histogram into 'Histogram' sheet.")
            
            # Write Token Data sheet
            percentiles_df.to_excel(writer, sheet_name='Token Data', index=False, startrow=0)
            percentile_chunks_df.to_excel(writer, sheet_name='Token Data', index=False, startrow=len(percentiles_df) + 2)
            logging.info(f"Wrote 'Token Data' sheet to {analysis_excel}")
        
        # Optionally remove the histogram image
        if os.path.exists(histogram_img):
            os.remove(histogram_img)
            logging.info(f"Removed temporary histogram image {histogram_img}")
    except Exception as e:
        logging.error(f"Error generating Excel report {analysis_excel}: {e}")

def process_directory(task: dict, common_params: dict, encoding):
    """Process a single directory based on the provided task configuration."""
    input_directory = Path(task["input_directory"])
    chunked_file = Path(task["chunked_file"])
    analysis_file = Path(task["analysis_file"])
    header_level = task["header_level"]
    
    # Populate headers
    headers_to_split_on = populate_headers_to_split_on(header_level)
    
    # Loader kwargs
    loader_kwargs = {
        "autodetect_encoding": common_params.get("autodetect_encoding", True),
        "file_glob": common_params.get("file_glob", "**/*.md")
    }
    
    # Load and concatenate text
    text_concatenated = load_and_concatenate_text(input_directory, TextLoader, loader_kwargs)
    if not text_concatenated:
        logging.warning(f"Skipping directory {input_directory} due to loading issues.")
        return
    
    # Split text
    md_header_splits = split_text(text_concatenated, headers_to_split_on)
    if not md_header_splits:
        logging.warning(f"No splits created for directory {input_directory}.")
        return
    
    # Process chunks
    documents = process_chunks(md_header_splits, encoding, header_level)
    if not documents:
        logging.warning(f"No documents processed for directory {input_directory}.")
        return
    
    # Write chunked Markdown
    write_chunked_markdown(documents, chunked_file)
    
    # Create DataFrame
    df = pd.DataFrame.from_dict(documents, orient='index')
    
    # Analyze tokens
    percentiles_df, percentile_chunks_df = analyze_tokens(df, common_params["token_thresholds"])
    
    # Generate Excel report
    generate_excel_report(
        df,
        percentiles_df,
        percentile_chunks_df,
        analysis_file,
        common_params["histogram_image"],
        common_params["histogram_bins"],
        common_params["token_thresholds"]
    )

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process multiple input directories based on a configuration file.')
    parser.add_argument('config', type=str, help='Path to the configuration JSON file.')
    args = parser.parse_args()
    
    config_path = Path(args.config)
    
    # Load configuration
    config = load_config(config_path)
    
    processing_tasks = config.get("processing_tasks", [])
    common_params = config.get("common_parameters", {})
    
    if not processing_tasks:
        logging.error("No processing tasks found in the configuration.")
        sys.exit(1)
    
    # Setup logging
    log_directory = common_params.get("log_directory")
    if log_directory:
        setup_logging(log_directory)
    else:
        # If no log directory is specified, set up logging only to console
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info("No log directory specified. Logging only to console.")
    
    # Initialize encoding once
    encoding_name = common_params.get("encoding_name", "cl100k_base")
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        logging.info(f"Initialized encoding: {encoding_name}")
    except Exception as e:
        logging.error(f"Error initializing encoding '{encoding_name}': {e}")
        sys.exit(1)
    
    # Loop through each processing task
    for idx, task in enumerate(processing_tasks, start=1):
        logging.info(f"Starting processing task {idx}/{len(processing_tasks)}")
        process_directory(task, common_params, encoding)
        logging.info(f"Completed processing task {idx}/{len(processing_tasks)}")
    
    logging.info("All processing tasks have been completed successfully.")

if __name__ == "__main__":
    main()
