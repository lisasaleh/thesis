import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import warnings
from dataclasses import dataclass
from typing import Dict, Any, Optional
import traceback


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class APIConfig:
    base_url: str
    model_name: str
    api_key: str = "EMPTY"
    max_tokens: int = 512
    temperature: float = 0.0
    timeout: float = 120.0
    retries: int = 3
    backoff: float = 2.0


def _chat_completions_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/v1"):
        return f"{base_url}/chat/completions"
    return f"{base_url}/v1/chat/completions"


def _extract_usage(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    usage = data.get("usage")
    return usage if isinstance(usage, dict) else None


def call_model(
    prompt: str,
    system_prompt: str = None,
    *,
    backend: str = "local",
    local_llm: "LocalLLM" = None,
    api_config: APIConfig = None,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> str:
    """
    Shared model call wrapper for local HuggingFace inference and vLLM's
    OpenAI-compatible chat completions API.
    """
    if backend == "local":
        if local_llm is None:
            raise ValueError("local backend requires local_llm")
        return local_llm.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )

    if backend != "api":
        raise ValueError(f"Unknown backend: {backend}")
    if api_config is None:
        raise ValueError("api backend requires api_config")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": api_config.model_name,
        "messages": messages,
        "max_tokens": max_tokens if max_tokens is not None else api_config.max_tokens,
        "temperature": temperature if temperature is not None else api_config.temperature,
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_config.api_key or 'EMPTY'}",
    }
    url = _chat_completions_url(api_config.base_url)

    last_error = None
    for attempt in range(1, api_config.retries + 1):
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=api_config.timeout) as response:
                response_body = response.read().decode("utf-8")
            data = json.loads(response_body)
            usage = _extract_usage(data)
            if usage:
                print(f"[DEBUG] API token usage: {usage}", file=sys.stderr, flush=True)
            choices = data.get("choices", [])
            if not choices:
                raise RuntimeError(f"API response did not include choices: {response_body[:500]}")
            message = choices[0].get("message", {})
            content = message.get("content", "")
            return str(content).strip()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_error = e
            if attempt >= api_config.retries:
                raise RuntimeError(
                    f"API server unreachable at {url} after {api_config.retries} attempts: {e}"
                ) from e
        except Exception as e:
            last_error = e
            if attempt >= api_config.retries:
                raise RuntimeError(f"API call failed after {api_config.retries} attempts: {e}") from e

        sleep_for = api_config.backoff * (2 ** (attempt - 1))
        print(
            f"[WARN] API call failed on attempt {attempt}/{api_config.retries}: {last_error}; retrying in {sleep_for:.1f}s",
            file=sys.stderr,
            flush=True,
        )
        time.sleep(sleep_for)

    raise RuntimeError(f"API call failed: {last_error}")


class APILLM:
    def __init__(self, config: APIConfig):
        self.config = config
        self.model_name = config.model_name

    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        return call_model(
            prompt=prompt,
            system_prompt=system_prompt,
            backend="api",
            api_config=self.config,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )


def add_backend_args(parser):
    parser.add_argument("--backend", choices=["local", "api"], default=os.environ.get("LLM_BACKEND", "local"))
    parser.add_argument("--api_base_url", type=str, default=os.environ.get("LLM_API_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--api_model_name", type=str, default=os.environ.get("LLM_API_MODEL_NAME"))
    parser.add_argument("--api_key", type=str, default=os.environ.get("LLM_API_KEY", "EMPTY"))
    parser.add_argument("--api_max_tokens", type=int, default=int(os.environ.get("LLM_API_MAX_TOKENS", "1200")))
    parser.add_argument("--api_temperature", type=float, default=float(os.environ.get("LLM_API_TEMPERATURE", "0")))
    parser.add_argument("--api_timeout", type=float, default=float(os.environ.get("LLM_API_TIMEOUT", "120")))
    parser.add_argument("--api_retries", type=int, default=int(os.environ.get("LLM_API_RETRIES", "3")))
    parser.add_argument("--api_backoff", type=float, default=float(os.environ.get("LLM_API_BACKOFF", "2")))


def create_llm_from_args(args):
    if args.backend == "api":
        model_name = args.api_model_name or args.model_name
        print(
            f"[DEBUG] Using API backend | base_url={args.api_base_url} | model={model_name}",
            flush=True,
        )
        return APILLM(APIConfig(
            base_url=args.api_base_url,
            model_name=model_name,
            api_key=args.api_key,
            max_tokens=args.api_max_tokens,
            temperature=args.api_temperature,
            timeout=args.api_timeout,
            retries=args.api_retries,
            backoff=args.api_backoff,
        ))

    print("[DEBUG] Starting model load...", flush=True)
    llm = LocalLLM(args.model_name)
    print("[DEBUG] Model load finished.", flush=True)
    return llm

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

        # Decide whether to load from local files only.
        local_only = (
            os.path.exists(model_name)
            or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
            or os.environ.get("FORCE_LOCAL_ONLY") == "1"
        )

        # Use trust_remote_code to allow custom local code; safe when loading local artifacts.
        trust_remote = True

        if local_only:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, local_files_only=True, trust_remote_code=trust_remote
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=trust_remote, **token_args
            )

        print(
            f"[DEBUG] Loading model for: {model_name} | cuda={torch.cuda.is_available()}",
            file=sys.stderr,
            flush=True,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")

        try:
            # Load model with local_files_only when appropriate to avoid contacting HF hub on compute nodes.
            model_kwargs = {
                "device_map": "auto",
                "dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                "trust_remote_code": True,
            }

            if local_only:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, local_files_only=True, **model_kwargs
                )
            else:
                # pass token args when not local-only (e.g., access token)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, **model_kwargs, **token_args
                )
        except Exception as e:
            print(f"[ERROR] from_pretrained failed: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            traceback.print_exc()
            raise

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.tokenizer.padding_side = 'left'  # Required for decoder-only models with batched generation

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
        
        # With left-padding, we need to track where each prompt actually starts and ends
        # to properly extract only the generated tokens (not padding or input)
        input_lens = (model_inputs["input_ids"] != self.tokenizer.pad_token_id).sum(dim=1)
        
        print(
            f"[DEBUG] Batch generation start | batch_size={len(prompts)} | avg_input_tokens={input_lens.float().mean():.0f}",
            file=sys.stderr,
            flush=True,
        )
        
        with torch.no_grad():
            # Suppress warning about unused generation kwargs (top_p, top_k)
            # They're only used when do_sample=True
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*generation flags are not valid.*")
                output_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=(temperature > 0),
                    temperature=temperature if temperature > 0 else None,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
        
        results = []
        for i in range(len(prompts)):
            # Find where the actual prompt ends (accounting for left-padding)
            # First, find where first non-padding token starts
            first_real_idx = (model_inputs["input_ids"][i] != self.tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
            if len(first_real_idx) > 0:
                start_idx = first_real_idx[0].item()
                # Prompt ends at: where it started + its length
                end_idx = start_idx + input_lens[i].item()
            else:
                # Fallback if all padding (shouldn't happen)
                end_idx = input_lens[i].item()
            
            # Extract only the generated portion
            generated_ids = output_ids[i][end_idx:]
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
            # Suppress warning about unused generation kwargs (top_p, top_k)
            # They're only used when do_sample=True
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*generation flags are not valid.*")
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

    # Attempt 3: escape control characters in strings
    escaped = candidate
    # Replace unescaped newlines/tabs inside string values with escaped versions
    escaped = re.sub(r'([^\\])\n', r'\1\\n', escaped)
    escaped = re.sub(r'([^\\])\t', r'\1\\t', escaped)
    escaped = re.sub(r'([^\\])\r', r'\1\\r', escaped)
    
    try:
        return json.loads(escaped)
    except json.JSONDecodeError:
        pass

    # Attempt 4: repair via model (if LLM available)
    if llm is not None:
        # Check if JSON looks truncated (ends with incomplete structure)
        is_truncated = candidate.count('"') % 2 == 1 or not candidate.rstrip().endswith('}')
        
        repair_prompt = f"""
Maak van onderstaande tekst geldige JSON. Het kan onvolledig zijn.

Regels:
- Geef ALLEEN geldige JSON terug, niets anders.
- Voeg geen uitleg toe.
- Verander de betekenis niet.
- Behoud exact dezelfde velden.
- Sluit alle haken en accolades correct af.
- Escape alle newlines en speciale tekens in strings.

Tekst:
{candidate}
""".strip()

        try:
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
            except json.JSONDecodeError:
                pass
        except Exception:
            pass
    
    # Attempt 5: If all else fails and JSON is truncated, return empty structure
    is_truncated = candidate.count('"') % 2 == 1 or not candidate.rstrip().endswith('}')
    if is_truncated:
        import sys
        print(f"[WARN] JSON truncated, returning empty result for: {candidate[:100]}...", file=sys.stderr)
        return {"claims": []}

    raise ValueError(f"JSON parsing mislukt:\n{candidate}")


def generate_json(
    llm,
    prompt: str,
    system_prompt: str = None,
    max_new_tokens: int = 300,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    raw, parsed = generate_json_with_raw(
        llm=llm,
        prompt=prompt,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    return parsed


def generate_json_with_raw(
    llm,
    prompt: str,
    system_prompt: str = None,
    max_new_tokens: int = 300,
    temperature: float = 0.0,
) -> tuple[str, Dict[str, Any]]:
    raw = llm.generate(
        prompt=prompt,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    print("[DEBUG] Raw generation received", file=sys.stderr, flush=True)

    parsed = extract_json_with_repair(raw, llm=llm)
    return raw, parsed
