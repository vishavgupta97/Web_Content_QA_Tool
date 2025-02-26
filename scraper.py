import gradio as gr
import requests
from bs4 import BeautifulSoup
import re
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline

# ==========================
# 🔹 INITIALIZING COMPONENTS
# ==========================

# Define the directory for ChromaDB storage
persist_directory = "./chroma_db"
os.makedirs(persist_directory, exist_ok=True)  # Ensure the directory exists

# Initialize the embedding model for text similarity search
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize Chroma vector store with persistent storage
vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# Load the Hugging Face question-answering model
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")


# ==========================
# 🔹 URL VALIDATION FUNCTION
# ==========================

def is_valid_url(url):
    """Check if the provided URL is a valid web address."""
    regex = re.compile(
        r'^(https?:\/\/)?'  # Matches http:// or https://
        r'(([A-Za-z0-9-]+\.)+[A-Za-z]{2,6})'  # Matches domain
        r'(\/.*)?$', re.IGNORECASE  # Matches optional path
    )
    return re.match(regex, url) is not None


# ==========================
# 🔹 WEB SCRAPING FUNCTION
# ==========================

def scrape_and_store(urls):
    """Scrapes text from multiple URLs and stores it in ChromaDB."""
    urls = [url.strip() for url in urls.split(";") if url.strip()]  # Split and clean URL input
    success_urls = []
    failed_urls = []

    for url in urls:
        if not is_valid_url(url):
            failed_urls.append(url)
            continue

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Check for HTTP request errors
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract text content from various HTML tags
            elements = []
            for tag in ["h1", "h2", "h3", "p", "li"]:
                for element in soup.find_all(tag):
                    text = element.get_text().strip()
                    if text:
                        elements.append(text)

            if not elements:
                failed_urls.append(url)
                continue

            content = "\n".join(elements)  # Combine extracted text

            # Split content into smaller chunks for better retrieval
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            documents = text_splitter.create_documents([content])

            # Attach metadata (source URL) to each document
            for doc in documents:
                doc.metadata = {"source": url}

            # Store extracted text embeddings in ChromaDB
            vector_store.add_documents(documents)
            success_urls.append(url)
        except requests.exceptions.RequestException:
            failed_urls.append(url)

    # Return ingestion status messages
    success_msg = f"Successfully ingested: {', '.join(success_urls)}" if success_urls else ""
    failed_msg = f"Failed to ingest: {', '.join(failed_urls)}" if failed_urls else ""
    return success_msg + "\n" + failed_msg if success_msg or failed_msg else "No valid URLs provided."


# ==========================
# 🔹 QUESTION-ANSWERING FUNCTION
# ==========================

def answer_question(question):
    """Retrieves relevant content from ChromaDB and answers the question."""
    try:
        if not vector_store._collection.count():  # Check if the database has data
            return "No content available. Please ingest a webpage first."

        # Retrieve relevant documents based on similarity search
        retrieved_docs = vector_store.similarity_search(question, k=5)
        if not retrieved_docs:
            return "No relevant content found."

        # Create context from retrieved documents
        context = " ".join([doc.page_content for doc in retrieved_docs])

        # Use the Hugging Face model to generate an answer
        result = qa_pipeline(question=question, context=context)
        return result["answer"]
    except Exception as e:
        return f"Error processing question: {e}"


# ==========================
# 🔹 GRADIO USER INTERFACE
# ==========================

with gr.Blocks(css="""
    body { height: 100vh; overflow-y: auto; background-color: #f4f4f4; font-family: Arial, sans-serif; }
    .gradio-container { max-width: 100%; height: 100%; margin: auto; padding: 20px; background: white; border-radius: 10px; box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1); overflow-y: auto; }
    .title { text-align: center; font-size: 24px; font-weight: bold; color: #333; }
    .instructions { text-align: center; font-size: 16px; color: #666; margin-bottom: 20px; }
""") as demo:
    gr.Markdown("""
    <div class='title'>🌐 Web Content Q&A Tool</div>
    <div class='instructions'>
    1️⃣ Enter one or more webpage URLs (separated by `;`).<br>
    2️⃣ Click **'Ingest Content'** to process the webpages.<br>
    3️⃣ Ask a question based on the ingested data.<br>
    4️⃣ Get an answer strictly based on the webpage content.<br>
    </div>
    """)

    with gr.Column():
        url_input = gr.Textbox(label="🔗 Enter URLs (separated by ';')",
                               placeholder="https://example.com; https://another.com")
        scrape_button = gr.Button("📥 Ingest Content", variant="primary")
        scrape_output = gr.Textbox(label="Status", interactive=False, lines=3)

        question_input = gr.Textbox(label="❓ Ask a Question", placeholder="What is the main topic of the article?")
        ask_button = gr.Button("🤖 Get Answer", variant="primary")
        answer_output = gr.Textbox(label="📌 Answer", interactive=False, lines=5)

    # Connect buttons to functions
    scrape_button.click(scrape_and_store, inputs=[url_input], outputs=[scrape_output])
    ask_button.click(answer_question, inputs=[question_input], outputs=[answer_output])

# Launch the Gradio app
demo.launch()
