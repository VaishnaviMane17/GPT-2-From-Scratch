import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple
from functools import partial
import tiktoken

class GPTDatasetV1(Dataset):
    """
    Dataset for GPT-like models, creating input-target pairs with sliding windows.
    """
    def __init__(self, txt: str, tokenizer: tiktoken.Encoding, max_length: int, stride: int):
        self.input_ids: List[torch.Tensor] = []
        self.target_ids: List[torch.Tensor] = []
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.target_ids[idx]


class InstructionDataset(Dataset):
    """
    Dataset for instruction-response pairs, pre-tokenizing full texts.
    """
    def __init__(self, data: List[Dict], tokenizer: tiktoken.Encoding):
        self.data = data
        self.encoded_texts: List[List[int]] = []
        for entry in data:
            instruction_plus_input = format_input(entry)  # Assuming format_input is imported from utils
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> List[int]:
        return self.encoded_texts[index]


def custom_collate_fn(
    batch: List[List[int]],
    pad_token_id: int = 50256,
    ignore_index: int = -100,
    allowed_max_length: int = None,
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Custom collate function for padding and preparing inputs/targets."""
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]
        inputs_lst.append(inputs)
        targets_lst.append(targets)
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor


def create_dataloader_v1(
    txt: str,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """Create DataLoader for GPT-like datasets."""
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )


def create_instruction_dataloader(
    data: List[Dict],
    tokenizer: tiktoken.Encoding,
    batch_size: int = 8,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
    device: str = "cpu",
    allowed_max_length: int = 1024
) -> DataLoader:
    """Create DataLoader for instruction datasets with custom collation."""
    dataset = InstructionDataset(data, tokenizer)
    customized_collate_fn = partial(
        custom_collate_fn, device=device, allowed_max_length=allowed_max_length
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )