import json
import os
import re
from typing import Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class LocalLLM:
    def __init__(self, model_name: str):
        self.model_name = model_name

        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt: str, max_new_tokens: int = 300, temperature: float = 0.0) -> str:
        messages = [{"role": "user", "content": prompt}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        model_inputs = self.tokenizer(
            [text],
            return_tensors="pt",
        )

        model_inputs = {k: v.to(self.model.device) for k, v in model_inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        input_length = model_inputs["input_ids"].shape[1]
        generated_ids = output_ids[0][input_length:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return text.strip()


def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()

    # Remove markdown fences
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # Try full parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting first JSON object
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in output:\n{text}")

    candidate = match.group(0)

    # First try candidate as-is
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Simple cleanup: remove trailing commas before ] or }
    cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse model output as JSON.\nOriginal output:\n{text}\n\nCandidate:\n{candidate}\n\nCleaned:\n{cleaned}\n\nError: {e}")