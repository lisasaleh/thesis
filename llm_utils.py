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

def extract_json_with_repair(text: str, llm=None) -> dict:
    text = text.strip()

    # Remove markdown fences
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # Extract JSON candidate
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found:\n{text}")

    candidate = match.group(0)

    # First attempt
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # 🔥 SECOND ATTEMPT: repair with LLM
    if llm is not None:
        repair_prompt = f"""
Fix the following text so that it is valid JSON.

Rules:
- Return ONLY valid JSON.
- Do not change the meaning.
- Do not add or remove fields.

Text:
{candidate}
"""

        repaired = llm.generate(
            prompt=repair_prompt,
            max_new_tokens=300,
            temperature=0.0,
        )

        repaired = repaired.strip()

        try:
            return json.loads(repaired)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Repair failed.\nOriginal:\n{candidate}\n\nRepaired:\n{repaired}\n\nError: {e}"
            )

    # If no repair possible
    raise ValueError(f"JSON parsing failed:\n{candidate}")