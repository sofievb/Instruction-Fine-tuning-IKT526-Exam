from config import SEED, GRADIENT_ACCUMULATION_STEPS, NUM_EPOCHS
from tqdm import tqdm
import torch
import math
from utils import set_seed, save_json
import time
from torch.cuda.amp import autocast, GradScaler
import os
set_seed(SEED)

def train_epoch(model, dataloader, optimizer, scheduler, device,
                grad_accum_steps, scaler=None, use_amp=False):
    """
    Train for one epoch.
    Returns:
        average_loss: float
    """
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    progress = tqdm(enumerate(dataloader), total=len(dataloader),
                    desc="Training", ncols=100)

    first_batch = True
    for step, batch in progress:
        if first_batch:
            print("[TRAIN] First batch shapes:")
            print("  input_ids:", batch["input_ids"].shape)
            print("  attention_mask:", batch["attention_mask"].shape)
            print("  labels:", batch["labels"].shape)
            first_batch = False

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with autocast(enabled=use_amp, dtype=torch.float16):

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            raw_loss = outputs.loss
            loss = raw_loss / grad_accum_steps

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        else:
            loss.backward()
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        total_loss += raw_loss.item()
        progress.set_postfix({"batch_loss": f"{loss.item():.4f}"})

    progress.close()
    avg_loss = total_loss / len(dataloader)
    print(f"[TRAIN] Epoch finished with avg_loss={avg_loss:.4f}")

    return avg_loss


def validate(model, dataloader, device, use_amp=False):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast(enabled=use_amp, dtype=torch.float16):

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    return avg_loss, perplexity


def train_model(model, train_dl, val_dl, optimizer, scheduler, device):
    train_losses = []
    val_ppls = []
    val_losses = []
    os.makedirs("outputs", exist_ok=True)

    
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    best_val_loss = float("inf")

    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    start_total_time = time.time()
    for epoch in range(NUM_EPOCHS):  
        print(f"=== Epoch {epoch + 1} / {NUM_EPOCHS} ===")
        start_time = time.time()

        avg_train_loss = train_epoch(
            model=model,
            dataloader=train_dl,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            grad_accum_steps=GRADIENT_ACCUMULATION_STEPS,  
            scaler=scaler,
            use_amp=use_amp,
        )

        val_loss, val_ppl = validate(model, val_dl, device, use_amp=use_amp)

        elapsed = time.time() - start_time
        print(
            f"[EPOCH {epoch+1}] train_loss={avg_train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, val_ppl={val_ppl:.2f},"
            f"time={elapsed/60:.2f} min"
        )


        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        val_ppls.append(val_ppl)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained("outputs/best_lora_model")
            print(f"Best model saved with val_loss={val_loss:.4f}")
            torch.save(model.state_dict(), "outputs/best_model.pt")
    total_elapsed_time = time.time() - start_total_time
    print(f"Total training time {total_elapsed_time}\n")

    if device.type == "cuda":
        peak_mem_bytes = torch.cuda.max_memory_allocated(device)
        peak_mem_gb = peak_mem_bytes / (1024 ** 3)
        print(f"[TRAIN] Peak GPU memory: {peak_mem_gb:.2f} GB")
    else:
        peak_mem_gb = None
        print("[TRAIN] Peak GPU memory: N/A (running on CPU)")

    history = {
        "epochs": [
            {
                "epoch": i+1,
                "train_loss": train_losses[i],
                "val_loss": val_losses[i],
                "val_ppl": val_ppls[i],
            }
            for i in range(len(train_losses))
        ],
        "total_time_sec": total_elapsed_time,
        "peak_memory_gb": peak_mem_gb,
    }
    save_json(history, "outputs/training_history.json")


