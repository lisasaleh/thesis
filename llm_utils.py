import json
import os
import re
import sys
from typing import Dict, Any
import traceback


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class LocalLLM:
    def __init__(self, model_name: str):
        self.model_name = model_name

        print(f"[DEBUG] Loading tokenizer for model: {model_name}", file=sys.stderr, flush=True)

        hf_token = (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        )

        if hf_token is None:
            token_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "../hf_token.txt")
            )
            if os.path.exists(token_path):
                with open(token_path, "r") as f:
                    hf_token = f.read().strip()

        print(f"[DEBUG] HF token loaded: {hf_token is not None}", file=sys.stderr, flush=True)

        token_args = {"token": hf_token} if hf_token else {}

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **token_args)

        print(
            f"[DEBUG] Loading model for: {model_name} | cuda={torch.cuda.is_available()}",
            file=sys.stderr,
            flush=True,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                **token_args
            )
        except Exception as e:
            print(f"[ERROR] from_pretrained failed: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            traceback.print_exc()
            raise

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("[DEBUG] Model + tokenizer ready", file=sys.stderr, flush=True)

    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0
    ) -> str:
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        print(
            f"[DEBUG] Prompt tokenization start | prompt_chars={len(prompt)}",
            file=sys.stderr,
            flush=True,
        )

        model_inputs = self.tokenizer([text], return_tensors="pt")
        input_device = next(self.model.parameters()).device
        model_inputs = {k: v.to(input_device) for k, v in model_inputs.items()}

        input_len = model_inputs["input_ids"].shape[1]

        print(
            f"[DEBUG] Generation start | input_tokens={input_len} | max_new_tokens={max_new_tokens}",
            file=sys.stderr,
            flush=True,
        )

        with torch.no_grad():
            output_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0][input_len:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        print("[DEBUG] Generation finished", file=sys.stderr, flush=True)
        return text.strip()

    def batch_generate(
        self,
        prompts: list,
        system_prompt: str = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0
    ) -> list:
        """Process multiple prompts in a batch for efficiency."""
        messages_list = []
        
        for prompt in prompts:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            messages_list.append(messages)
        
        # Tokenize all prompts
        texts = [
            self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            for msgs in messages_list
        ]
        
        print(
            f"[DEBUG] Batch tokenization start | batch_size={len(prompts)} | avg_chars={sum(len(p) for p in prompts) // len(prompts)}",
            file=sys.stderr,
            flush=True,
        )
        
        model_inputs = self.tokenizer(texts, return_tensors="pt", padding=True)
        input_device = next(self.model.parameters()).device
        model_inputs = {k: v.to(input_device) for k, v in model_inputs.items()}
        
        input_lens = (model_inputs["input_ids"] != self.tokenizer.pad_token_id).sum(dim=1)
        
        print(
            f"[DEBUG] Batch generation start | batch_size={len(prompts)} | avg_input_tokens={input_lens.float().mean():.0f}",
            file=sys.stderr,
            flush=True,
        )
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        results = []
        for i, input_len in enumerate(input_lens):
            generated_ids = output_ids[i][input_len:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            results.append(text.strip())
        
        print("[DEBUG] Batch generation finished", file=sys.stderr, flush=True)
        return results
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        print(
            f"[DEBUG] Prompt tokenization start | prompt_chars={len(prompt)}",
            file=sys.stderr,
            flush=True,
        )

        model_inputs = self.tokenizer([text], return_tensors="pt")
        model_inputs = {k: v.to(self.model.device) for k, v in model_inputs.items()}

        input_len = model_inputs["input_ids"].shape[1]

        print(
            f"[DEBUG] Generation start | input_tokens={input_len} | max_new_tokens={max_new_tokens}",
            file=sys.stderr,
            flush=True,
        )

        with torch.no_grad():
            output_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0][input_len:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        print("[DEBUG] Generation finished", file=sys.stderr, flush=True)
        return text.strip()


# ============================
# JSON utilities
# ============================
def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_json_with_repair(text: str, llm: LocalLLM = None) -> Dict[str, Any]:
    text = _strip_fences(text)

    match = re.search(r"\{.*", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Geen JSON-object gevonden:\n{text}")

    candidate = match.group(0).strip()

    # Attempt 1: direct parse
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Attempt 2: remove trailing commas
    normalized = re.sub(r",\s*([}\]])", r"\1", candidate)

    try:
        return json.loads(normalized)
    except json.JSONDecodeError:
        pass

    # Attempt 3: repair via model
    if llm is not None:
        repair_prompt = f"""
Maak van onderstaande tekst geldige JSON.

Regels:
- Geef alleen geldige JSON terug.
- Voeg geen uitleg toe.
- Verander de betekenis niet.
- Behoud exact dezelfde velden.
- Sluit alle haken en accolades correct af.

Tekst:
{candidate}
""".strip()

        repaired = llm.generate(
            prompt=repair_prompt,
            max_new_tokens=512,
            temperature=0.0,
        )

        repaired = _strip_fences(repaired)

        match_repaired = re.search(r"\{.*", repaired, flags=re.DOTALL)
        if match_repaired:
            repaired = match_repaired.group(0).strip()

        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)

        try:
            return json.loads(repaired)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"JSON-reparatie mislukt.\nOrigineel:\n{candidate}\n\nGerepareerd:\n{repaired}\n\nError: {e}"
            )

    raise ValueError(f"JSON parsing mislukt:\n{candidate}")


def generate_json(
    llm: LocalLLM,
    prompt: str,
    system_prompt: str = None,
    max_new_tokens: int = 300,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    raw = llm.generate(
        prompt=prompt,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    print("[DEBUG] Raw generation received", file=sys.stderr, flush=True)

    parsed = extract_json_with_repair(raw, llm=llm)
    return parsed