MODEL_ID = "meta-llama/Llama-3.2-1B"
MAX_LENGTH = 512

SUBSET = 14000

TRAIN_SIZE = 10000
VAL_SIZE = 2000
TEST_SIZE = 2000

SEED = 42

# ========= Training Hyperparams =========
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
EPSILON = 1e-8
NUM_EPOCHS = 3
GRADIENT_ACCUMULATION_STEPS = 2
WARMUP_RATIO = 0.05   
BATCH_SIZE = 16

# -------- LoRA Params --------
LORA_R = 16 
LORA_ALPHA = LORA_R*2
LORA_DROPOUT = 0
TARGET_MODULES = ["q_proj", "k_proj", 
                  "v_proj", "o_proj"]


# ========= Sampling =========
TOP_P = 0.9
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 128