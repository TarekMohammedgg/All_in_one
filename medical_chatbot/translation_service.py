import json
import re
from pydantic import BaseModel, Field
from typing import Dict, List, Any  # Added missing imports

class TranslateModel(BaseModel):
    Translated_conversation: str = Field(..., min_length=5, description="The conversation translated from Arabic into English")

class TranslationService:
    @staticmethod
    def create_translate_template(ar_text: str) -> List[Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": "\n".join([
                    "You are a professional translator.",
                    "You will be provided with an Arabic text.",
                    "Translate the text into English.",
                    "Extract JSON details according to the Pydantic schema provided.",
                    "Make sure that the values in the JSON are written in *English*.",
                    "Respond ONLY with a valid JSON object that follows the schema - no markdown fences, no extra text."
                ])
            },
            {
                "role": "user",
                "content": "\n".join([
                    "## conversation",
                    ar_text,
                    "",
                    "## Pydantic Details",
                    json.dumps(TranslateModel.model_json_schema(), ensure_ascii=False),
                    "",
                    "## Translation : ",
                    "```json"
                ])
            }
        ]

    @staticmethod
    def generate_translation(message: List[Dict[str, str]], tokenizer, model, device: str, max_new_tokens: int = 1500) -> Dict[str, Any]:
        input_ids = tokenizer.apply_chat_template(
            message,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)

        gen_tokens = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
        )

        gen_tokens = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(input_ids, gen_tokens)
        ]

        gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
        match = re.search(r'\{.*?\}', gen_text, re.DOTALL)
        if not match:
            raise ValueError("No JSON block found in model output")

        json_block = match.group(0).strip()
        return json.loads(json_block)
