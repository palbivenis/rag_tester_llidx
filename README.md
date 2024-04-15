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
- 