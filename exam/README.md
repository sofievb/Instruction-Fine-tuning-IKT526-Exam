# Fine-tuning Llama-3.2-1B using LoRA adapters on Stanford Alpaca Dataset
This repository contains an implementation for instruction fine-tuning the
meta-llama/Llama-3.2-1B model using LoRA adapters on the
yahma/alpaca-cleaned dataset, developed as part of the IKT526 course final project.

The project uses a custom PyTorch training loop to evaluate the fine-tuned model using perplexity and token-level F1 score, as required by the assignment.
## Dataset:
- Dataset: yahma/alpaca-cleaned
- Source: HuggingFace Datasets
- Task: Instruction fine-tuning (instruction + optional input → response)

## Model & LoRA Configuration

Base model: meta-llama/Llama-3.2-1B
Fine-tuning method: LoRA (PEFT)
LoRA parameters: 
- r (rank): 16
- alpha: r * 2
- target modules: ['q_proj', 'k_proj', 'v_proj','o_proj']

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


## Repository
- `train.py`: Training and validating model
- `evaluate.py`: 
- `main.py`: Main file for training, evaluation and inference.
    1. Loading and initializing Llama base model and model with LoRA ranks, initializing scheduler, optimizer
    2. Running evaluation and inference scripts
    
- `inference.py`: Script for inference on base vs. fine-tuned model and sampling strategies comparison
- `config.py`: Defining training hyperparameters, LoRA parameters, model type
- `outputs/`: 
    - `/best_lora_model/`: saved adapter configurations
    - `/generations/`: results generated in `inference.py` and `config.py`
- `requirements.txt`:  Python dependencies
## Evaluation Metrics
- Perplexity: Measures how well the model predicts a sample. Lower perplexity indicates better performance.
- Token-level F1 Score: Evaluates the accuracy of the model's generated tokens against the reference tokens, balancing precision and recall.

### Notes
The LoRA adapter is saved separately and can be loaded on top of the base model.

The project intentionally avoids transformers.Trainer to demonstrate a custom training loop.

Code is structured for clarity, modularity, and reproducibility.