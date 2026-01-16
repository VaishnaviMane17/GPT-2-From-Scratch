import json
from tqdm import tqdm
from src.generation import generate, text_to_token_ids, token_ids_to_text
from src.utils import format_input  # From utils.py
from typing import List, Dict
import urllib.request
import psutil

def check_ollama_running() -> bool:
    """Check if Ollama is running."""
    for proc in psutil.process_iter(["name"]):
        if "ollama" in proc.info["name"]:
            return True
    return False

def query_model(prompt: str, model: str = "llama3", url: str = "http://localhost:11434/api/chat") -> str:
    """Query Ollama model via API."""
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"seed": 123, "temperature": 0, "num_ctx": 2048}
    }
    payload = json.dumps(data).encode("utf-8")
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")
    response_data = ""
    with urllib.request.urlopen(request) as response:
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json["message"]["content"]
    return response_data

def extract_responses(model: GPTModel, test_data: List[Dict], tokenizer: tiktoken.Encoding, device: str, file_path: str = "instruction-data-with-response.json"):
    """Extract and save model responses for test data."""
    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
        input_text = format_input(entry)
        token_ids = generate(model, text_to_token_ids(input_text, tokenizer).to(device), 256, model.config["context_length"], 50256)
        generated_text = token_ids_to_text(token_ids, tokenizer)
        response_text = generated_text[len(input_text):].replace("### Response:", "").strip()
        test_data[i]["model_response"] = response_text
    with open(file_path, "w") as file:
        json.dump(test_data, file, indent=4)

def generate_model_scores(json_data: List[Dict], json_key: str = "model_response", model: str = "llama3") -> List[int]:
    """Score model responses using Ollama."""
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry[json_key]}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
            f"Respond with the integer number only."
        )
        score = query_model(prompt, model)
        try:
            scores.append(int(score))
        except ValueError:
            print(f"Could not convert score: {score}")
    return scores