### Project: PDF Content Retrieval and Query System Using Langchain and Cassandra

This repository contains a project that demonstrates how to build a PDF content retrieval and query system using the Langchain library and Cassandra. The main objective is to extract text from a PDF document, store it in a vector database, and enable querying using natural language.

#### Key Features
- **PDF Text Extraction**: Extracts text content from a PDF file.
- **Text Splitting**: Splits the extracted text into manageable chunks suitable for vector storage.
- **Vector Storage**: Uses Cassandra to store text embeddings.
- **Natural Language Querying**: Allows querying the stored text using OpenAI's language model.

#### Dependencies
- `cassio`
- `datasets`
- `langchain`
- `openai`
- `tiktoken`
- `PyPDF2`

#### Setup

1. **Install the required packages**:
    ```bash
    pip install -q cassio datasets langchain openai tiktoken PyPDF2
    ```

2. **Configure API Keys and Tokens**:
    - Set the `ASTRA_DB_APPLICATION_TOKEN` and `ASTRA_DB_ID` for Cassandra.
    - Set the `OpenAI_API_KEY` for OpenAI's language model.

#### Code Overview

1. **Imports and Setup**:
    ```python
    from langchain.vectorstores.cassandra import Cassandra
    from langchain.indexes.vectorstore import VectorStoreIndexWrapper
    from langchain.llms import OpenAI
    from langchain.embeddings import OpenAIEmbeddings
    from datasets import load_dataset
    import cassio
    from PyPDF2 import PdfReader
    ```

2. **PDF Text Extraction**:
    ```python
    PdfReader = PdfReader('budget_speech.pdf')
    
    raw_text = ''
    for i, page in enumerate(PdfReader.pages):
        content = page.extract_text()
        if content:
            raw_text += content
    ```

3. **Initialize Cassandra**:
    ```python
    cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)
    ```

4. **Create Langchain Embeddings and LLM Objects**:
    ```python
    LLM = OpenAI(openai_api_key=OpenAI_API_KEY)
    embedding = OpenAIEmbeddings(openai_api_key=OpenAI_API_KEY)
    ```

5. **Store Text Chunks in Cassandra**:
    ```python
    from langchain.text_splitter import CharacterTextSplitter

    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=800, chunk_overlap=200, length_function=len)
    texts = text_splitter.split_text(raw_text)
    
    astra_vector_store = Cassandra(embedding=embedding, table_name="qa_mini_demo", session=None, keyspace=None)
    astra_vector_store.add_texts(texts[:50])
    ```

6. **Query System**:
    ```python
    astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)
    
    first_question = True
    while True:
        if first_question:
            query_text = input("\nEnter your question (or type 'quit' to exit): ").strip()
        else:
            query_text = input("\nWhat's your next question (or type 'quit' to exit): ").strip()
        if query_text.lower() == "quit":
            break
        if query_text == "":
            continue

        first_question = False
        print("\nQuestion: \"%s\"" % query_text)

        answer = astra_vector_index.query(query_text, llm=LLM).strip()
        print("Answer: \"%s\"\n" % answer)

        print("First documents by relevance:")
        for doc, score in astra_vector_store.similarity_search_with_score(query_text, k=4):
            print(" [%0.4f] \"%s....\"" % (score, doc.page_content[:84]))
    ```

#### Usage
1. Run the script.
2. Enter your questions when prompted. Type 'quit' to exit the query loop.

This project showcases the integration of PDF text extraction, vector storage with Cassandra, and natural language querying using Langchain and OpenAI's LLM. Feel free to customize and extend this code to fit your specific use case.
