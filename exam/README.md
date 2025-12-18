# Instruction Fine-Tuning Llama-3.2-1B with LoRA (IKT526 Final Project)
This repository contains an implementation for instruction fine-tuning the
meta-llama/Llama-3.2-1B model using LoRA adapters on the
yahma/alpaca-cleaned dataset, developed as part of the IKT526 course final project.

The project uses a custom PyTorch training loop to evaluate the fine-tuned model using perplexity and token-level F1 score, as required by the assignment.
## Dataset:
- Dataset: yahma/alpaca-cleaned
- Source: HuggingFace Datasets
- Task: Instruction fine-tuning (instruction + optional input → response)

## Model & LoRA Configuration

- Base model: meta-llama/Llama-3.2-1B
- Fine-tuning method: LoRA (PEFT)

**LoRA parameters**
- Rank (r): 16
- Alpha: 2 × r
- Target modules: q_proj, k_proj, v_proj, o_proj

## Training Setup
Epochs: 3
Optimizer: AdamW
Scheduler: Linear with warmup
Gradient accumulation: enabled
Mixed precision: Automatic Mixed Precision (AMP) when using CUDA
Best model checkpoint saved based on validation loss
Training statistics (loss, perplexity, runtime, peak GPU memory) are logged and
stored in outputs/training_history.json.
## Running the Code
1. Install dependencies:
pip install -r requirements.txt
2. Train the model:
python main.py
3. Run inference:
python inference.py
4. Evaluate on test set:
python evaluate.py


## Repository Structure

- train.py: Training and validation loops (custom PyTorch implementation)
- evaluate.py: Evaluation on test set (perplexity, token-level F1)
- inference.py: Inference on base vs fine-tuned model and sampling strategy comparison
- main.py:
  - Loads model and LoRA adapters
  - Initializes optimizer and scheduler
  - Runs training, evaluation, and inference
- config.py: Training hyperparameters, LoRA configuration, model settings
- outputs/:
  - best_lora_model/: saved LoRA adapter weights
  - generations/: generated outputs and evaluation results
- requirements.txt: Python dependencies

## Evaluation Metrics
- Perplexity: Measures how well the model predicts a sample. Lower perplexity indicates better performance.
- Token-level F1 Score: Evaluates the accuracy of the model's generated tokens against the reference tokens, balancing precision and recall.

## Disclaimer
This repository is intended for educational and portfolio purposes.
Large model checkpoints are intentionally excluded from version control.

The project intentionally avoids transformers.Trainer to demonstrate a custom training loop.

Code is structured for clarity, modularity, and reproducibility.
