from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import PreTrainedTokenizer, PreTrainedModel
import re
from typing import Dict
from medical_chatbot.medical_session import MedicalSession
from langchain.prompts import PromptTemplate
from medical_chatbot.translation_service import TranslationService
from summarization_task.summarization import create_summary_template, generate_summary
from config import FEW_SHOT_PROMPT


def create_app(
    cohere_model: PreTrainedModel,
    cohere_tokenizer: PreTrainedTokenizer,
    qa_llm,
    retriever
) -> FastAPI:
    app = FastAPI()
    sessions: Dict[str, MedicalSession] = {}

    # --- Summarization ---
    class SummarizationInput(BaseModel):
        Translated_conversation: str

    @app.post("/summarize")
    def summarize(input_data: SummarizationInput):
        try:
            en_text = input_data.Translated_conversation
            summary_prompt = create_summary_template(en_text, cohere_model)
            summary_output = generate_summary(summary_prompt, cohere_tokenizer, cohere_model)
            return summary_output
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # --- RAG Chatbot ---
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

    # --- Session-Based Medical Chatbot ---
    class ChatRequest(BaseModel):
        message: str

    @app.post("/chat/{session_id}")
    def chat_with_doctor(session_id: str, request: ChatRequest):
        if session_id not in sessions:
            sessions[session_id] = MedicalSession()
        session = sessions[session_id]
        response = session.process_input(request.message, cohere_tokenizer, cohere_model, cohere_model.device)
        return {
            "response": response,
            "finished": str(session.finished),
            "full_conversation": session.messages
        }

    @app.get("/export/{session_id}")
    def export_session(session_id: str):
        if session_id not in sessions:
            return JSONResponse(status_code=404, content={"error": "الجلسة غير موجودة."})
        session = sessions[session_id]
        ar_text = session.get_full_arabic_conversation()
        try:
            message = TranslationService.create_translate_template(ar_text)
            translated_json = TranslationService.generate_translation(
                message, cohere_tokenizer, cohere_model, cohere_model.device)
        except Exception as e:
            translated_json = {"error": f"فشل في الترجمة: {str(e)}"}
        return {
            "session_id": session_id,
            "finished": str(session.finished),
            "full_conversation": session.messages,
            "translated_conversation": translated_json
        }

    @app.get("/translate/{session_id}")
    def translate_conversation(session_id: str):
        if session_id not in sessions:
            return JSONResponse(status_code=404, content={"error": "الجلسة غير موجودة."})
        session = sessions[session_id]
        ar_text = session.get_full_arabic_conversation()
        try:
            message = TranslationService.create_translate_template(ar_text)
            translated_json = TranslationService.generate_translation(
                message, cohere_tokenizer, cohere_model, cohere_model.device)
            return translated_json
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"فشل في الترجمة: {str(e)}"})

    @app.get("/")
    def root():
        return {"message": "Summarization + RAG + Session-based Chatbot API is running."}

    return app
