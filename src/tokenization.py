import re
import tiktoken
from typing import List, Dict

class SimpleTokenizerV1:
    """
    A simple tokenizer that splits text into tokens and maps them to integer IDs.
    Handles basic punctuation and whitespace.
    """
    def __init__(self, vocab: Dict[str, int]):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> List[int]:
        """Encode text into token IDs."""
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to text."""
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


class SimpleTokenizerV2:
    """
    An improved tokenizer that handles unknown words with <unk> and <endoftext>.
    """
    def __init__(self, vocab: Dict[str, int]):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> List[int]:
        """Encode text, replacing unknown tokens with <unk>."""
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to text."""
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text


def get_bpe_tokenizer(encoding: str = "gpt2") -> tiktoken.Encoding:
    """Load a BPE tokenizer (e.g., for GPT-2)."""
    return tiktoken.get_encoding(encoding)


def build_vocab_from_text(raw_text: str) -> Dict[str, int]:
    """Build a vocabulary from preprocessed text."""
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    all_tokens = sorted(set(preprocessed))
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    vocab = {token: i for i, token in enumerate(all_tokens)}
    return vocab