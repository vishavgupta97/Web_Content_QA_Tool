import gradio as gr
from duckduckgo_search import DDGS
from transformers import pipeline

# Load the pre-trained question-answering model
qa_pipeline = pipeline("question-answering")


def retrieve_url_content(url):
    """Retrieve content from a URL using DuckDuckGo."""
    try:
        results = DDGS(url)
        if results:
            return results[0]["body"]
        else:
            return "No content found for this URL."
    except Exception as e:
        return f"Error retrieving content: {str(e)}"


def ask_question(url, question):
    """Answer questions based on the retrieved content."""
    if not url or not question:
        return "Please provide both a URL and a question."

    # Retrieve content from the URL
    context = retrieve_url_content(url)
    if "Error" in context or "No content" in context:
        return context

    # Use the QA model to answer the question
    result = qa_pipeline(question=question, context=context)
    return result["answer"]


# Gradio Interface
def chatbot_interface(url, question, chat_history):
    """Gradio chatbot interface."""
    if not url:
        return chat_history, "Please enter a URL first."

    # Add user question to chat history
    chat_history.append((question, ""))

    # Get the answer
    answer = ask_question(url, question)

    # Update chat history with the answer
    chat_history[-1] = (question, answer)

    return chat_history, ""


# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# Web Content Q&A Tool")
    with gr.Row():
        url_input = gr.Textbox(label="Enter URL", placeholder="https://example.com")
        chatbot = gr.Chatbot(label="Chat")
        question_input = gr.Textbox(label="Ask a Question", placeholder="Type your question here")
        clear_button = gr.Button("Clear")

        # Define interactions
        question_input.submit(
            chatbot_interface,
            inputs=[url_input, question_input, chatbot],
            outputs=[chatbot, question_input]
        )
        clear_button.click(lambda: [], None, chatbot, queue=False)

# Launch the app
demo.launch()