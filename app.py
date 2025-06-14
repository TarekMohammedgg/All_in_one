from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import PreTrainedTokenizer, PreTrainedModel
import re
from langchain.prompts import PromptTemplate
from summarization_task.summarization import create_summary_template, generate_summary
from config import FEW_SHOT_PROMPT
from langchain.chains import RetrievalQA

def create_app(
    summary_model: PreTrainedModel,
    summary_tokenizer: PreTrainedTokenizer,
    qa_llm,  # from qa_chain.combine_documents_chain.llm_chain.llm
    retriever  # from qa_chain.retriever
) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI()

    # --- Summarization Endpoint ---
    class SummarizationInput(BaseModel):
        Translated_conversation: str

    @app.post("/summarize")
    def summarize(input_data: SummarizationInput):
        try:
            en_text = input_data.Translated_conversation
            summary_prompt = create_summary_template(en_text, summary_model)
            summary_output = generate_summary(summary_prompt, summary_tokenizer, summary_model)
            return summary_output
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # --- Chatbot Endpoint ---
    class DoctorQuestion(BaseModel):
        message: str
        translated_conversation: str

    template = PromptTemplate(
        template=FEW_SHOT_PROMPT,
        input_variables=["translated_conversation", "raag_reference", "question"]
    )

    def clean_text(text: str) -> str:
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        text = ansi_escape.sub('', text).strip()
        text = re.sub(r'Prompt after formatting:.*?\n', '', text, flags=re.DOTALL)
        match = re.search(r'Answer:\s*(.*)', text, re.DOTALL)
        answer = match.group(1).strip() if match else text
        return "\n".join(dict.fromkeys(answer.split("\n")))

    @app.post("/ask")
    def ask(msg: DoctorQuestion):
        user_input = msg.message.strip()
        translated_conversation = msg.translated_conversation.strip()

        try:
            retrieved_docs = retriever.get_relevant_documents(user_input)
            raag_context = "\n".join([doc.page_content for doc in retrieved_docs])

            prompt = template.format(
                translated_conversation=translated_conversation,
                raag_reference=raag_context,
                question=user_input
            )

            result = qa_llm(prompt)
            cleaned = clean_text(result)
            return {"response": cleaned}
        except Exception as e:
            return {"error": str(e)}

    @app.get("/")
    def root():
        return {"message": "Summarization + RAG Doctor Assistant API is running."}

    return app
