import gradio as gr
import scrapy
from scrapy.crawler import CrawlerProcess
import re
import faiss
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

try:
    import torch

    if not torch.cuda.is_available():
        print("Warning: CUDA not available, running on CPU.")

    model_name = "mistralai/Mistral-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    qa_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
    llm = HuggingFacePipeline(pipeline=qa_pipeline)
except ImportError:
    llm = None
    print("Error: Required dependencies are not installed. Please install PyTorch and Transformers.")

# Initialize FAISS Vector DB
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS(embedding_model, dimension=384)  # Adjust based on embedding dimension

ingested_data = {}


class WebSpider(scrapy.Spider):
    name = "web_spider"
    start_urls = []

    def __init__(self, url=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_urls = [url]

    def parse(self, response):
        page_text = " ".join(response.xpath('//p//text()').getall())
        ingested_data[self.start_urls[0]] = page_text

        # Split text into chunks and store in vector DB
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_text(page_text)
        vector_store.add_texts(chunks)


def is_valid_url(url):
    """Validate the URL format."""
    return bool(re.match(r'^(https?:\/\/)?[\w.-]+(?:\.[\w\.-]+)+[/#?]?.*$', url))


def scrape_and_store(url):
    """Scrapes text from the given URL and stores it."""
    if not is_valid_url(url):
        return "Invalid URL format. Please enter a valid URL."

    process = CrawlerProcess(settings={
        "LOG_LEVEL": "ERROR",
    })
    process.crawl(WebSpider, url=url)
    process.start()

    if url in ingested_data:
        return f"Content from {url} ingested successfully!"
    return "Failed to fetch content. The site may block scraping. Try another URL."


def answer_question(url, question):
    """Answers questions using RAG (retrieval-augmented generation)."""
    if url not in ingested_data:
        return "URL not ingested. Please scrape content first."

    if not question.strip():
        return "Please enter a valid question."

    retriever = vector_store.as_retriever()
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    try:
        response = rag_chain.run(question)
        return response
    except Exception:
        return "Unable to generate a response. Try rephrasing your question."


# Gradio UI
def main():
    with gr.Blocks() as demo:
        gr.Markdown("## Web Content Q&A Tool with RAG + Vector DB")

        with gr.Row():
            url_input = gr.Textbox(label="Enter URL")
            scrape_button = gr.Button("Ingest Content")
        scrape_output = gr.Textbox(label="Scraping Status", interactive=False)

        with gr.Row():
            question_input = gr.Textbox(label="Ask a Question", max_lines=2)
            ask_button = gr.Button("Get Answer")
        answer_output = gr.Textbox(label="Answer", interactive=False)

        scrape_button.click(scrape_and_store, inputs=[url_input], outputs=[scrape_output])
        ask_button.click(answer_question, inputs=[url_input, question_input], outputs=[answer_output])

    demo.launch()


if __name__ == "__main__":
    main()
