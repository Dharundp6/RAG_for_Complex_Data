import os
import textwrap
from pathlib import Path
import asyncio
import PyPDF2
import markdown

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from llama_parse import LlamaParse

# Set API keys
GROQ_API_KEY = ""
LLAMA_PARSE_KEY = ""

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

def print_response(response):
    response_txt = response["result"]
    for chunk in response_txt.split("\n"):
        if not chunk:
            print()
            continue
        print("\n".join(textwrap.wrap(chunk, 100, break_long_words=False)))

instruction = """The provided document is NVIDIA's First Quarter Fiscal 2024 Financial Results.
This is a press release that provides detailed financial information about NVIDIA's performance for the first quarter of its fiscal year 2024.
It includes unaudited financial statements, management's commentary, highlights of key developments, and disclosures related to NVIDIA's outlook for the next quarter.
The document contains many financial tables and figures. Try to be precise while answering questions based on the information in this press release."""

async def main():
    parser = LlamaParse(
        api_key=LLAMA_PARSE_KEY,
        result_type="markdown",
        parsing_instruction=instruction,
        max_timeout=5000,
    )

    llama_parse_documents = await parser.aload_data("NVIDIAAn.pdf")
    parsed_doc = llama_parse_documents[0].text

    # Create the data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Write the parsed content to the parsed_document.md file
    document_path = data_dir / "parsed_document.md"
    with document_path.open("w", encoding="utf-8") as f:
        f.write(parsed_doc)

    # Load the parsed_document.md file
    loader = UnstructuredMarkdownLoader(document_path)
    loaded_documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=128)
    docs = text_splitter.split_documents(loaded_documents)

    # Download embeddings
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

    qdrant = Qdrant.from_documents(
        docs,
        embeddings,
        path="./db",
        collection_name="document_embeddings",
    )

    retriever = qdrant.as_retriever(search_kwargs={"k": 5})

    compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2")
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )

    llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")

    prompt_template = """
    Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}

    Answer the question and provide additional helpful information,
    based on the pieces of information, if applicable. Be succinct.

    Responses should be properly formatted to be easily read.
    """

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=compression_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt, "verbose": False},
    )

  

    query = "Compare the Gross profit from 2022 and 2023?"
    response = qa.invoke(query)
    print(markdown.markdown(response["result"]))

asyncio.run(main())
