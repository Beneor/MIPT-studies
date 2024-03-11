# Importing necessary libraries

import locale
def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding

import os
import sys
import random
import accelerate
import faiss
from tqdm.auto import tqdm, trange
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import torch
import transformers
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
from transformers import TrainingArguments, pipeline

# Change working directory
os.chdir('D:/sokrat_bot_generative')

# Parameters

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)


from trl import SFTTrainer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
from transformers import TrainingArguments, pipeline
from datasets import Dataset

# QLoRA parameters

# LoRA attention dimension
lora_r = 64
# Alpha parameter for LoRA scaling
lora_alpha = 16
# Dropout probability for LoRA layers
lora_dropout = 0.1
# bitsandbytes parameters
# Activate 4-bit precision base model loading
use_4bit = True
# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"
# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"
# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

# SFT parameters

# Maximum sequence length to use
max_seq_length = None
# Pack multiple short examples in the same input sequence to increase efficiency
packing = False
# Load the entire model on the GPU 0
device_map = {"": 0}

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

llm_model.config.use_cache = True
torch.cuda.empty_cache()

# Loading pretrained model and adapters

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)
model = PeftModel.from_pretrained(base_model, 'D:/sokrat_bot_generative/checkpoint-500')
model = model.merge_and_unload()

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Inference functions

def get_completion(query, model, tokenizer):
    
    """Returns the generated answer """
    
    prompt = 'Ниже дан вопрос, дай на него краткий ответ на русском языке\n\n'
    prompt += f'### Вопрос:\n{query}\n'
    prompt += f'### Ответ:\n'
    encodeds = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    model_inputs = encodeds.to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(**model_inputs, max_new_tokens=70, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    answer = ''
    answer_ind = 0
    answer_2ind = 0
    answer_2 = ''
    answer_3 = ''
    lst = []
    
    for i in decoded[0]:
        lst.append(i.strip('/n'))
    answer = ''.join(lst)
    if ('### Ответ:') in answer:
        answer_ind = answer.find('### Ответ:')
        answer_2 = answer[answer_ind+11:]
        if '### Вопрос:' in answer_2:
            answer_2ind = answer_2.find('### Вопрос:')
            if '.' in answer_2:
                end = answer_2.find('/n')
            elif '?' in answer_2:
                end = answer_2.find('?')
            elif '!' in answer_2:
                end = answer_2.find('!')
            else:
                end = len(answer_2)-1
            answer_3 = answer_2[answer_2ind+7:end+1]
        else:
            answer_3 = answer_2
    else:
      answer_3 = answer
    return answer_3

def final_result(query):

    """Returns the generated answer for Gradio"""
    
    answer_final = get_completion(query, model=llm_model, tokenizer=llm_tokenizer)
    return answer_final

import gradio as gr

demo = gr.Interface(
    fn=final_result,
    inputs=["text"],
    outputs=["text"],
)

demo.launch()
