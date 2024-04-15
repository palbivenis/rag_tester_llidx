### Rag Tester using Llamaindex RAG framework ###

Conduct evaluations of rag strategies against various data sets

#### Strategies supported ####
Examine the effects of the changing parameters on various RAG metrics (correctness, faithfulness etc.)
- Basic Rag
  - Chunk Size, Reranking
- Sentence Window
  - Window Size, Reranking
- Recursive Rag
  - Chunk Size, Reranking
- Fusion Retriever
  - Weights assign to vector search vs. keyword search

#### Data sets ####
- Available under datasets. For each data set
- files -> contains the document(s) that will be ingested
- questions -> evaluation questions (along with expected answers). In spreadsheet and json format
- evaluations/llamaindex -> contains results of evaluation runs

#### Running Evaluations ####
- Make the following updates to config.py
    - Enter your API keys
    - Choose the data set for evaluation (directory, file, questions, quick test question)
    - Choose LLM (for embedding and generation)
    - Set RAG_STRATEGY to appropriate value
        - Refer to the relevant notebook for permitted values
    - Set any adiitonal parameters relevant to the strategy
- Run notebook

#### Evaluation Output ####
- Each notebook run will create an xlsx file under evaluations/llamaindex/data with the name
- \<data_set\>\_\<strategy\>\_GM\_\<generation LLM\>\_EM\_\<embedding LLM\>\_\<parameters\>\_\<run date\>.xlsx
- analysis folder contains Tableau files that provide analysis across runs
- the .twbx file is a read-only file that can be viewed by the free [Tableau Reader](https://www.tableau.com/products/reader)
