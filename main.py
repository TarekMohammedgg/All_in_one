from fastapi import FastAPI
import nest_asyncio
from chatbot.embeddings import setup_qa_chain
from pyngrok import ngrok
import uvicorn
import os
from config import BASE_MODEL_ID, SUMMARIZATION_HF_TOKEN, NGROK_PORT, NGROK_AUTH_TOKEN
from model_loader import load_quantized_model
from app import create_app
from utils.utils import clear_gpu_cache

def main():
    nest_asyncio.apply()
    clear_gpu_cache()

    # Load RAG components
    qa_chain = setup_qa_chain()
    llm = qa_chain.combine_documents_chain.llm_chain.llm
    retriever = qa_chain.retriever

    # Tokens
    summary_hf_token = os.environ.get("HF_TOKEN", SUMMARIZATION_HF_TOKEN)
    ngrok_token = os.environ.get("NGROK_AUTH_TOKEN", NGROK_AUTH_TOKEN)

    if not summary_hf_token or not ngrok_token:
        raise ValueError("Missing HF_TOKEN or NGROK_AUTH_TOKEN")

    # Load summarization model
    model, tokenizer = load_quantized_model(
        model_name=BASE_MODEL_ID,
        auth_token=summary_hf_token
    )
    print(f"Model loaded on device: {model.device}")

    # Set up ngrok
    ngrok.set_auth_token(ngrok_token)
    for tunnel in ngrok.get_tunnels():
        ngrok.disconnect(tunnel.public_url)
    public_url = ngrok.connect(NGROK_PORT)
    print("🚀 Your API is live at:", public_url)

    # Run app
    app: FastAPI = create_app(model, tokenizer, llm, retriever)
    uvicorn.run(app, host="0.0.0.0", port=NGROK_PORT)

if __name__ == "__main__":
    main()
