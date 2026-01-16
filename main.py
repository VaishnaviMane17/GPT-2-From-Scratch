import torch
import tiktoken
from src.utils import download_and_load_file, format_input
from src.tokenization import get_bpe_tokenizer, build_vocab_from_text
from src.data_loading import create_dataloader_v1, create_instruction_dataloader
from src.model import load_pretrained_gpt
from src.training import train_model_simple, calc_loss_loader
from src.evaluation import extract_responses, generate_model_scores
from src.generation import text_to_token_ids, token_ids_to_text, generate

# Configs and Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHOOSE_MODEL = "gpt2-medium (355M)"
INSTRUCTION_URL = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
TEXT_FILE = "data/the-verdict.txt"

# Step 1: Load Data
data = download_and_load_file("instruction-data.json", INSTRUCTION_URL)
train_portion = int(len(data) * 0.85)
test_portion = int(len(data) * 0.1)
val_portion = len(data) - train_portion - test_portion
train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]

tokenizer = get_bpe_tokenizer()

# Step 2: Create DataLoaders
with open(TEXT_FILE, "r", encoding="utf-8") as f:
    raw_text = f.read()

train_loader = create_dataloader_v1(raw_text, batch_size=8, max_length=256, stride=128)
val_loader = create_dataloader_v1(raw_text, batch_size=8, max_length=256, stride=128, shuffle=False)  # Example, adjust for instruction data

instruction_train_loader = create_instruction_dataloader(train_data, tokenizer, device=DEVICE)
instruction_val_loader = create_instruction_dataloader(val_data, tokenizer, device=DEVICE, shuffle=False)
instruction_test_loader = create_instruction_dataloader(test_data, tokenizer, device=DEVICE, shuffle=False)

# Step 3: Load Model
model = load_pretrained_gpt(CHOOSE_MODEL)
model.to(DEVICE)

# Step 4: Train (Example with instruction data)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)
train_losses, val_losses, tokens_seen = train_model_simple(
    model, instruction_train_loader, instruction_val_loader, optimizer, DEVICE,
    num_epochs=1, eval_freq=5, eval_iter=5,
    start_context=format_input(val_data[0]), tokenizer=tokenizer
)

# Step 5: Evaluate
extract_responses(model, test_data, tokenizer, DEVICE)
scores = generate_model_scores(test_data)

print("Average Score:", sum(scores) / len(scores) if scores else 0)