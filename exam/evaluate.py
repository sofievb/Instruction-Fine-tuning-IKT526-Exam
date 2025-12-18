from utils import load_model, f1_score, sampling, save_json, read_json
import matplotlib.pyplot as plt
import os 
import numpy as np
from data import load_data
from collections import Counter
import torch
import re
from config import MAX_LENGTH, MAX_NEW_TOKENS
import random
def visualize_training(train_losses, val_ppls, val_losses):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, marker='o', label='Training Loss')
    plt.plot(epochs, val_losses, marker='o', label='Validation Loss')
    plt.grid(True)

    plt.xlabel('Epochs')
    plt.xticks(list(epochs))
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_ppls, marker='o', label='Validation Perplexity', color='orange')
    plt.grid(True)

    plt.xticks(list(epochs))
    plt.ylabel('Perplexity')
    plt.title('Validation Perplexity')

    plt.legend()

    plt.tight_layout()
    
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/training_validation_curves.png", dpi=300, bbox_inches="tight")
    plt.show()

def normalize_text(text: str):
    text = text.strip().lower()
    if not text:
        return []
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    return text.split()

def text_perplexity(model, tokenizer, text, device):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(device)
    input_ids = enc["input_ids"]
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss
    return float(torch.exp(loss).item())

def token_f1_score(pred, true):
    
    pred_tokens = normalize_text(pred)
    true_tokens = normalize_text(true)

    if len(pred_tokens) == 0 and len(true_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(true_tokens) == 0:
        return 0.0

    
    common_tokens = Counter(pred_tokens) & Counter(true_tokens)
    num_common = sum(common_tokens.values())
    if num_common == 0:
        return 0.0
    
    precision = 1.0 * num_common / len(pred_tokens)
    recall = 1.0 * num_common / len(true_tokens)

    return f1_score(precision, recall)

def compute_metrics():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    base_model, ft_model, tokenizer, device = load_model(device)
    tokenizer.pad_token = tokenizer.eos_token


    _, _, test = load_data()

    base_f1s, ft_f1s = [], []
    base_ppls, ft_ppls = [], []
    results = []

    base_model.eval()
    ft_model.eval()

    data = sampling(test, n_samples=10)

    for idx, sample in enumerate(data):
        instr = sample["instruction"]
        inp   = sample["input"]
        ref   = sample["output"]

        if inp.strip():
            prompt = (
                f"Instruction:\n{instr}\n\n"
                f"Input:\n{inp}\n\n"
                f"Answer:\n"
            )
        else:
            prompt = (
                f"Instruction:\n{instr}\n\n"
                f"Answer:\n"
            )


        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        ).to(device)

        with torch.no_grad():
            base_ids = base_model.generate(
                **encoded,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )
            ft_ids = ft_model.generate(
                **encoded,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )

        base_out = tokenizer.decode(base_ids[0], skip_special_tokens=True)
        ft_out   = tokenizer.decode(ft_ids[0],   skip_special_tokens=True)

        base_f1 = token_f1_score(base_out, ref)
        ft_f1   = token_f1_score(ft_out,   ref)

        base_f1s.append(base_f1)
        ft_f1s.append(ft_f1)

        base_ppl = text_perplexity(base_model, tokenizer, base_out, device)
        ft_ppl   = text_perplexity(ft_model,   tokenizer, ft_out,   device)

        base_ppls.append(base_ppl)
        ft_ppls.append(ft_ppl)

        results.append({
            "example_id": idx,
            "instruction": instr,
            "input": inp,
            "reference": ref,
            "base_output": base_out,
            "ft_output": ft_out,
            "base_f1": base_f1,
            "ft_f1": ft_f1,
            "base_ppl": base_ppl,
            "ft_ppl": ft_ppl,
            #Qualitative scores for manual annotation
            "base_score": None,
            "ft_score": None,
        })

    summary = {
        "base": {
            "f1_mean": float(np.mean(base_f1s)),
            "f1_std":  float(np.std(base_f1s)),
            "ppl_mean": float(np.mean(base_ppls)),
            "ppl_std":  float(np.std(base_ppls)),
        },
        "ft": {
            "f1_mean": float(np.mean(ft_f1s)),
            "f1_std":  float(np.std(ft_f1s)),
            "ppl_mean": float(np.mean(ft_ppls)),
            "ppl_std":  float(np.std(ft_ppls)),
        },
    }

    payload = {
        "description": "Evaluation on 10 diverse test samples",
        "summary": summary,
        "examples": results,
    }
    save_json(payload, "outputs/generations/test_set_evaluation.json")
    
if __name__ == "__main__":
    
    training_history = read_json("outputs/training_history.json")
    
    train_losses = [e["train_loss"] for e in training_history["epochs"]]
    val_losses   = [e["val_loss"]  for e in training_history["epochs"]]
    val_ppls     = [e["val_ppl"]   for e in training_history["epochs"]]
    compute_metrics()
    visualize_training(train_losses, val_ppls, val_losses)
    
