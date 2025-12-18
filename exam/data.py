from config import SUBSET, TRAIN_SIZE, VAL_SIZE, TEST_SIZE, SEED, MODEL_ID, MAX_LENGTH, BATCH_SIZE

from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

def load_data():
    """
    Load alpaca
    Split 14k subset into train, validation, test [5:1:1]
    """
    ds = load_dataset("yahma/alpaca-cleaned")
    subset_dict = ds["train"].train_test_split(train_size=SUBSET, seed=SEED) # type: ignore
    subset = subset_dict["train"]

    train_test = subset.train_test_split(test_size=TEST_SIZE, seed=SEED)
    train_val = train_test["train"]
    test = train_test["test"]

    train_val_split = train_val.train_test_split(test_size=VAL_SIZE, seed=SEED)
    train = train_val_split["train"]
    val   = train_val_split["test"]
    print(train.column_names)

    return train, val, test

def format_text(example):
    
    instruction = example["instruction"]
    input_text  = example["input"]
    output_text = example["output"]

    if input_text and input_text.strip():
        prompt = (
            f"Instruction:\n{instruction}\n\n"
            f"Input:\n{input_text}\n\n"
            f"Answer:\n{output_text}"
        )
    else:
        prompt = (
            f"Instruction:\n{instruction}\n\n"
            f"Answer:\n{output_text}"
        )

    example["text"] = prompt
    return example


def tokenize(samples):
    return tokenizer(
        samples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

def tokenize_splits(train, val, test):
    
    train = train.map(format_text)
    val   = val.map(format_text)
    test  = test.map(format_text)

    print("[TOKENIZE] Columns before tokenization:", train.column_names)

    train = train.map(tokenize, batched=True)
    val   = val.map(tokenize, batched=True)
    test  = test.map(tokenize, batched=True)

    example = train[0]
    print("[TOKENIZE] Example keys:", example.keys())
    print("[TOKENIZE] Example input_ids len:", len(example["input_ids"]))
    print("[TOKENIZE] Example attention_mask len:", len(example["attention_mask"]))
    return train, val, test

def dataloaders(train, val, test):
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val, batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test, batch_size=BATCH_SIZE)
    return train_loader, val_loader, test_loader
