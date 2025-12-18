from train import train_model
from config import (MODEL_ID, LEARNING_RATE, WEIGHT_DECAY, EPSILON, 
                    WARMUP_RATIO, NUM_EPOCHS, GRADIENT_ACCUMULATION_STEPS, 
                    LORA_R, LORA_ALPHA, LORA_DROPOUT, TARGET_MODULES)
from data import load_data, tokenize_splits, dataloaders

from transformers import LlamaForCausalLM
from transformers import get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
import torch
import math
from utils import save_json

def count_params(model):
    total = sum(params.numel() for params in model.parameters())
    trainable = sum(params.numel() for params in model.parameters() if params.requires_grad)
    return total, trainable

def main():

    train_data, val_data, test_data  = load_data()
    train_data, val_data, test_data = tokenize_splits(train_data, val_data, test_data)

    train_data = train_data.map(lambda x: {"labels": x["input_ids"]})
    val_data   = val_data.map(lambda x: {"labels": x["input_ids"]})
    test_data = test_data.map(lambda x: {"labels": x["input_ids"]})

    train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    train_dl, val_dl, _ = dataloaders(train_data, val_data, test_data)
    print(f"[MAIN] Batches -> train: {len(train_dl)}, val: {len(val_dl)}")
    
    base_model = LlamaForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,   
    )
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable()

    
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)

    model.print_trainable_parameters()

    total_params, trainable_params = count_params(model)
    print(f"Total params: {total_params}")
    print(f"Trainable params (LoRA): {trainable_params}")

    model_stats = {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
    }   
    save_json(model_stats, "outputs/model_parameter_counts.json")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n Using device: {device}")
    if device.type == 'cuda':
        print(f"\n GPU {torch.cuda.get_device_name(0)}")
    else:
        print(f"Training on CPU")
    
    model.to(device)
    print(f"\n Using device: {device}")
    
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=LEARNING_RATE, 
                                  weight_decay=WEIGHT_DECAY, 
                                  eps=EPSILON)

    steps_per_epoch = math.ceil(len(train_dl) / GRADIENT_ACCUMULATION_STEPS)
    total_train_steps = steps_per_epoch * NUM_EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_train_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_train_steps,
    )
    
    train_model(model=model, 
                train_dl=train_dl, 
                val_dl=val_dl, 
                optimizer=optimizer, 
                scheduler=scheduler, 
                device=device,
            )

if __name__ == "__main__":
    main()