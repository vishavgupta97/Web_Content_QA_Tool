import gradio as gr
from duckduckgo_search import DDGS
import re

try:
    from transformers import pipeline
    import torch

    if not torch.cuda.is_available():
        print("Warning: CUDA not available, running on CPU.")
    qa_model = pipeline("question-answering", model="distilbert-base-cased")
except ImportError:
    qa_model = None
    print("Error: PyTorch or TensorFlow is not installed. Please install PyTorch using 'pip install torch'.")

# Temporary database to store scraped content
ingested_data = {}


def is_valid_url(url):
    """Validate the URL format."""
    return bool(re.match(r'^(https?:\/\/)?[\w.-]+(?:\.[\w\.-]+)+[/#?]?.*$', url))


def scrape_and_store(url):
    """Scrapes text from the given URL and stores it."""
    if not is_valid_url(url):
        return "Invalid URL format. Please enter a valid URL."

    with DDGS() as ddgs:
        results = ddgs.text(url, max_results=1)
        if results and "body" in results[0]:
            content = results[0]["body"]
            if content.strip():
                ingested_data[url] = content
                return f"Content from {url} ingested successfully!"
            else:
                return "No readable content found on the page."
    return "Failed to fetch content. The site may block scraping. Try another URL."


def answer_question(url, question):
    """Answers questions based only on ingested content."""
    if qa_model is None:
        return "Error: PyTorch or TensorFlow is not installed. Please install PyTorch using 'pip install torch'."

    if url not in ingested_data:
        return "URL not ingested. Please scrape content first."

    if not question.strip():
        return "Please enter a valid question."

    context = ingested_data[url]
    try:
        answer = qa_model(question=question, context=context)
        return answer.get("answer", "No valid answer found.")
    except Exception:
        return "Unable to generate a response. Try rephrasing your question."


# Gradio UI
def main():
    with gr.Blocks() as demo:
        gr.Markdown("## Web Content Q&A Tool")

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
