import torch
from src.model import GPTModel
from src.tokenization import get_bpe_tokenizer

def text_to_token_ids(text: str, tokenizer: tiktoken.Encoding) -> torch.Tensor:
    """Convert text to token IDs."""
    return torch.tensor(tokenizer.encode(text, allowed_special={"<|endoftext|>"})).unsqueeze(0)

def token_ids_to_text(token_ids: torch.Tensor, tokenizer: tiktoken.Encoding) -> str:
    """Convert token IDs back to text."""
    return tokenizer.decode(token_ids.squeeze(0).tolist())

def generate(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    eos_id: int = 50256
) -> torch.Tensor:
    """Generate new tokens using the model."""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
        if idx_next.item() == eos_id:
            break
    return idx

def generate_and_print_sample(model: GPTModel, tokenizer: tiktoken.Encoding, device: str, start_context: str):
    """Generate and print a sample text."""
    model.eval()
    token_ids = generate(model, text_to_token_ids(start_context, tokenizer).to(device), 50, model.config["context_length"], 50256)
    print(token_ids_to_text(token_ids, tokenizer))
    model.train()