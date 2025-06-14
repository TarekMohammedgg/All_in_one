from pydantic import BaseModel, Field
from typing import List
import json
import re
from transformers import PreTrainedTokenizer, PreTrainedModel

class SummaryModel(BaseModel):
    """Pydantic schema for summarization output."""
    Patient_symptoms: List[str] = Field(..., min_length=5, max_length=300, description="Key diagnosis symptoms mentioned")
    Symptom_location: str = Field(..., min_length=5, max_length=300, description="Affected area")
    Duration: str = Field(..., min_length=5, max_length=300, description="How long the symptoms have persisted")
    Symptom_progression: str = Field(..., min_length=5, max_length=300, description="How they have changed over time")
    Risk_factors: str = Field(..., min_length=5, max_length=300, description="Sun exposure, family history, etc.")

def create_summary_template(en_text: str, summary_model: BaseModel) -> List[dict]:
    """
    Create a chat-style prompt template for summarization.
    
    Args:
        en_text: Input text to summarize
        summary_model: Pydantic model defining the schema
    
    Returns:
        List of prompt dictionaries for the model
    """
    return [
        {
            "role": "system",
            "content": "\n".join([
                "You are an NLP data parser.",
                "You will be provided by a text associated with a Pydantic schema.",
                "Generate the output in the same language as the input.",
                "Extract JSON details from text according to the Pydantic schema.",
                "Extract details as mentioned in text.",
                "Do not generate any introduction or conclusion."
            ])
        },
        {
            "role": "user",
            "content": "\n".join([
                "## conversation",
                en_text,
                "",
                "## Pydantic Details",
                json.dumps(summary_model.model_json_schema(), ensure_ascii=False),
                "",
                "## Summarization : ",
                "```json"
            ])
        }
    ]

def generate_summary(message: List[dict], tokenizer: PreTrainedTokenizer, model: PreTrainedModel, max_new_tokens: int = 200) -> dict:
    """
    Generate a JSON summary from the input prompt.
    
    Args:
        message: Chat-style prompt
        tokenizer: Model tokenizer
        model: Pre-trained language model
        max_new_tokens: Maximum tokens to generate
    
    Returns:
        Parsed JSON summary
    """
    input_ids = tokenizer.apply_chat_template(
        message,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    
    gen_tokens = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
    )
    
    gen_tokens = [output_ids[len(input_ids):] for input_ids, output_ids in zip(input_ids, gen_tokens)]
    
    gen_text = tokenizer.decode(gen_tokens[0])
    gen_text = gen_text.replace("<|END_RESPONSE|>", "").replace("<|END_OF_TURN_TOKEN|>", "").replace("```", "").strip()
    
    match = re.search(r'\{.*?\}', gen_text, re.DOTALL)
    if not match:
        raise ValueError("No JSON block found in model output")
    
    json_block = match.group(0).strip()
    return json.loads(json_block)