# llm_setup.py
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    pipeline
)
from langchain.llms import HuggingFacePipeline
from config import Doctor_HF_TOKEN, DOCTOR_MODEL_NAME_ID
from huggingface_hub import login
import warnings

login(token=Doctor_HF_TOKEN)

def create_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

def load_model(model_name, tokenizer_name, bnb_config):
    n_gpus = torch.cuda.device_count()
    max_memory = f'{15000}MB'

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory={i: max_memory for i in range(n_gpus)},
        trust_remote_code=True,
        token=Doctor_HF_TOKEN
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def create_text_generation_pipeline(model, tokenizer):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch_dtype,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.7,
        repetition_penalty=1.2,
    )

def setup_llm():
    bnb_config = create_bnb_config()
    model, tokenizer = load_model(DOCTOR_MODEL_NAME_ID, DOCTOR_MODEL_NAME_ID, bnb_config)
    text_generation_pipeline = create_text_generation_pipeline(model, tokenizer)
    return HuggingFacePipeline(pipeline=text_generation_pipeline)

def load_quantized_model(
    model_name: str,
    load_in_4bit: bool = True,
    use_double_quant: bool = True,
    quant_type: str = "nf4",
    compute_dtype=torch.bfloat16,
    auth_token: str = None
):
    """
    Load a quantized model and its tokenizer with specified configuration.
    
    Args:
        model_name: Hugging Face model ID
        load_in_4bit: Whether to use 4-bit quantization
        use_double_quant: Whether to use double quantization
        quant_type: Quantization type (e.g., 'nf4')
        compute_dtype: Data type for computation
        auth_token: Hugging Face authentication token
    
    Returns:
        Tuple of (model, tokenizer)
    """
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.models.auto.tokenization_auto")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_use_double_quant=use_double_quant,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    
    n_gpus = torch.cuda.device_count()
    max_memory = {i: "40960MB" for i in range(n_gpus)}
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory=max_memory,
        token=auth_token
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=auth_token)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer