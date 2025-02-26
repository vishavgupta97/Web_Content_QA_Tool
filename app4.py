import gradio as gr
import requests
from bs4 import BeautifulSoup
import re
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from transformers import pipeline
import os

# Initialize ChromaDB
persist_directory = "./chroma_db"
os.makedirs(persist_directory, exist_ok=True)
vector_store = Chroma(persist_directory=persist_directory,
                      embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
retriever = vector_store.as_retriever()

# Load Hugging Face Q&A Model
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")


def is_valid_url(url):
    """Check if the provided URL is valid."""
    regex = re.compile(
        r'^(https?:\/\/)?'  # http:// or https://
        r'(([A-Za-z0-9-]+\.)+[A-Za-z]{2,6})'  # domain...
        r'(\/.*)?$', re.IGNORECASE)  # optional path
    return re.match(regex, url) is not None


def scrape_and_store(urls):
    """Scrapes text from multiple URLs and stores them in ChromaDB."""
    urls = [url.strip() for url in urls.split(";") if url.strip()]
    success_urls = []
    failed_urls = []

    for url in urls:
        if not is_valid_url(url):
            failed_urls.append(url)
            continue

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            paragraphs = [p.get_text() for p in soup.find_all("p")]
            content = "\n".join(paragraphs)

            if not content.strip():
                failed_urls.append(url)
                continue

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            documents = text_splitter.create_documents([content])

            # Add metadata (URL) to each document
            for doc in documents:
                doc.metadata = {"source": url}

            vector_store.add_documents(documents)
            success_urls.append(url)
        except requests.exceptions.RequestException:
            failed_urls.append(url)

    success_msg = f"Successfully ingested: {', '.join(success_urls)}" if success_urls else ""
    failed_msg = f"Failed to ingest: {', '.join(failed_urls)}" if failed_urls else ""
    return success_msg + "\n" + failed_msg if success_msg or failed_msg else "No valid URLs provided."


def answer_question(question, url=None):
    """Answers questions based on ingested content, automatically selecting the most relevant URL if none is provided."""
    try:
        if not vector_store._collection.count():
            return "No content available. Please ingest webpages first."

        retrieved_docs = retriever.get_relevant_documents(question)

        # If no URL is provided, infer the most relevant one
        if not url:
            url_counts = {}
            for doc in retrieved_docs:
                source_url = doc.metadata.get("source")
                url_counts[source_url] = url_counts.get(source_url, 0) + 1

            if url_counts:
                url = max(url_counts, key=url_counts.get)  # Select URL with most relevant docs
            else:
                return "No relevant content found."

        # Filter results by inferred or specified URL
        filtered_docs = [doc for doc in retrieved_docs if doc.metadata.get("source") == url]
        if not filtered_docs:
            return f"No relevant content found for {url}."

        context = " ".join([doc.page_content for doc in filtered_docs])
        result = qa_pipeline(question=question, context=context)
        return f"(From {url}) {result['answer']}"
    except Exception as e:
        return f"Error processing question: {e}"


# Gradio UI
def main():
    with gr.Blocks() as demo:
        gr.Markdown("## Web Content Q&A Tool")

        with gr.Row():
            url_input = gr.Textbox(label="Enter URLs (separated by ;)")
            scrape_button = gr.Button("Ingest Content")
        scrape_output = gr.Textbox(label="Scraping Status", interactive=False)

        with gr.Row():
            question_input = gr.Textbox(label="Ask a Question")
            url_select = gr.Textbox(label="Enter URL to Ask About (Optional)")
            ask_button = gr.Button("Get Answer")
        answer_output = gr.Textbox(label="Answer", interactive=False)

        scrape_button.click(scrape_and_store, inputs=[url_input], outputs=[scrape_output])
        ask_button.click(answer_question, inputs=[question_input, url_select], outputs=[answer_output])

    demo.launch()


if __name__ == "__main__":
    main()
