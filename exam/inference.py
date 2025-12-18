
import os
import torch
from utils import load_model, save_json, sampling
from data import load_data
from config import TOP_P, TEMPERATURE, MAX_NEW_TOKENS, SEED 


def format_prompt(sample):
    instr = sample["instruction"]
    inp   = sample["input"]
    if inp.strip():
        return (
            f"Instruction:\n{instr}\n\n"
            f"Input:\n{inp}\n\n"
            f"Answer:\n"
        )
    else:
        return (
            f"Instruction:\n{instr}\n\n"
            f"Answer:\n"
        )


def generate_base_vs_ft(base_model, ft_model, tokenizer, samples, device):

    prompts = [format_prompt(sample) for sample in samples]
    results = []
    for idx, (sample, prompt) in enumerate(zip(samples, prompts)):
        base_out = generate_strategic("greedy", tokenizer, base_model, device, prompt)
        ft_out   = generate_strategic("greedy", tokenizer, ft_model, device, prompt)

        results.append({
            "example_id": idx,
            "instruction": sample["instruction"],
            "input": sample["input"],
            "reference": sample["output"],
            "prompt": prompt,
            "base_output": base_out,
            "ft_output": ft_out,
        })

    save_json({"samples": results}, "outputs/generations/base_vs_ft_test.json")

def generate_strategic(strategy, tokenizer, model, device, prompt):
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    if strategy == "greedy":
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )
    elif strategy == "top_p":
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=1.0,   
                top_p=TOP_P,
            )
    elif strategy == "temperature":
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=1.0,     
            )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def compare_strategies(ft_model, tokenizer, device):


    prompts = [
                "Explain the concept of gradient descent.",
                "Create a short poem about winter",
                "Create a python for-loop",
                "What should i buy my mom for Christmas?",
                "Give me a review of Titanic?",
                "What is polynomial division? Explain step by step",
                "List three advantages of using databases.",
                "Summarize the importance of a well structured code",
                "Compare supervised and unsupervised learning in two sentences.",
                "Translate the following sentence into French: 'What should I have for dinner?'."
            ]
    results = []

    for idx, prompt in enumerate(prompts):
        greedy_out = generate_strategic("greedy", tokenizer, ft_model, device, prompt)
        top_p_out = generate_strategic("top_p", tokenizer, ft_model, device, prompt)
        temp_out  = generate_strategic("temperature", tokenizer, ft_model, device, prompt)

        results.append({
            "example_id": idx,
            "prompt": prompt,
            "greedy": greedy_out,
            "top_p": top_p_out,
            "temperature": temp_out,
        })

    save_json({"samples": results}, "outputs/generations/sampling_comparison.json")

def run_inference(n_samples=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model, ft_model, tokenizer, device = load_model(device)

    _, _, test = load_data()
    samples = sampling(test, n_samples)

    generate_base_vs_ft(base_model, ft_model, tokenizer, samples, device)

    compare_strategies(ft_model, tokenizer, device)

if __name__ == "__main__":
    run_inference(n_samples=10)
