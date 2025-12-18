import random
import numpy as np
import torch
from torch.nn import functional as F
from peft import PeftModel
from transformers import LlamaForCausalLM, AutoTokenizer
from config import MODEL_ID, SEED
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ADAPTER_DIR = os.path.join(BASE_DIR, "outputs", "best_lora_model")

def set_seed(seed):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

    base_model = LlamaForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    base_model.eval()

    ft_base = LlamaForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)

    ft_model = PeftModel.from_pretrained(
        ft_base,
        ADAPTER_DIR,
    ).to(device)
    ft_model.eval()

    return base_model, ft_model, tokenizer, device


def perplexity(logits, input_ids, attention_mask):
    
    logits = logits[:, :-1, :]
    labels = input_ids[:, 1:]
    attention_mask = attention_mask[:, 1:]

    log_probs = F.log_softmax(logits, dim=-1)
    target_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    target_log_probs = target_log_probs * attention_mask.to(log_probs.dtype)

    negative_log_likelihood = -target_log_probs.sum(dim=-1) / attention_mask.sum(dim=-1)

    perplexities = torch.exp(negative_log_likelihood)
    mean_perplexity = torch.mean(perplexities)

    return perplexities, mean_perplexity


def f1_score(prec, recall):
    if prec + recall == 0:
        return 0.0
    return float(2*(prec * recall) / (prec + recall))

def sampling(data, n_samples):
    
    random.seed(SEED)
    indices = random.sample(range(len(data)), n_samples)

    samples = [{"instruction": data[i]["instruction"],
                "input": data[i]["input"],
                "output": data[i]["output"]}
                for i in indices]

    return samples
def read_json(path):
    with open(path, "r") as f:
        return json.load(f)
        
def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

