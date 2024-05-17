import os
# from openai import OpenAI
# os.environ["OPENAI_API_KEY"] = "" # Enter your OpenAi Key
# client = OpenAI()
import gc
import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, GemmaTokenizer

# torch.backends.cuda.enable_mem_efficient_sdp(False)

tokenizer = GemmaTokenizer.from_pretrained("/kaggle/input/codegemma/transformers/7b-it/1")

_model = AutoModelForCausalLM.from_pretrained(
    "/kaggle/input/codegemma/transformers/7b-it/1",
    device_map="auto",
    torch_dtype="auto",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,  # Loading weights in 4-bit format
        bnb_4bit_quant_type="nf4",  # Using non-linear quantization with 4 bits
        bnb_4bit_compute_dtype=torch.bfloat16,  # Using bfloat16 for computation
        bnb_4bit_use_double_quant=True  # Using double quantization
    )
)
torch.backends.cuda.enable_mem_efficient_sdp(False)
def generate_from_thinker(prompts, max_tokens, model="gpt-4-1106-preview", temperature=0.7, n=1):
    instructions = """You are a thinker. I need you to help me think about some problems.
        You need to provide me the answer based on the format of the example."""
    message=instructions
    for i in range(len(prompts)): 
        message += prompts[i]["content"]
       
    tokens = tokenizer(message, return_tensors='pt').to(_model.device)
   
    outputs = _model.generate(
        **tokens,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=temperature,
     )
    context = tokenizer.decode(outputs[0][1:-1])
    return context[len(message):]    


def generate_from_judge(prompts, max_tokens, model="gpt-4-1106-preview", temperature=0.7, n=1):
    instructions = """You're a judge. I need you to make judgments on some statements."""
    message=instructions
    for i in range(len(prompts)): 
        message += prompts[i]["content"]
       
    tokens = tokenizer(message, return_tensors='pt').to(_model.device)
    outputs = _model.generate(
        **tokens,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=temperature,
     )
    context = tokenizer.decode(outputs[0][1:-1])
    return context[len(message):]   



def generate_from_excutor(prompts, max_tokens, model="gpt-4-1106-preview", temperature=0.7, n=1):
    instructions="""You're an excutor. I need you to calculate the final result based on some conditions and steps.
    You need to provide me the answer based on the format of the examples."""
    message=instructions
    for i in range(len(prompts)): 
        message += prompts[i]["content"]
    tokens = tokenizer(message, return_tensors='pt').to(_model.device)
    outputs = _model.generate(
        **tokens,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=temperature,
     )
    context = tokenizer.decode(outputs[0][1:-1])
    return context[len(message):]   

