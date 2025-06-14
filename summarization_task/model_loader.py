from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import warnings

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