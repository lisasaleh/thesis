import json
import os
import re
import sys
from typing import Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class LocalLLM:
    def __init__(self, model_name: str):
        self.model_name = model_name

        print(f"[DEBUG] Loading tokenizer for model: {model_name}", file=sys.stderr, flush=True)

        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        token_args = {"token": hf_token} if hf_token else {}

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **token_args)

        print(
            f"[DEBUG] Loading model for: {model_name} | cuda={torch.cuda.is_available()}",
            file=sys.stderr,
            flush=True,
        )

        # # ADD: 4-bit quantization config
        # quantization_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        #     bnb_4bit_use_double_quant=True,   # saves a bit of extra memory
        #     bnb_4bit_quant_type="nf4",        # best quality for 4-bit
        # )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            **token_args
        )

        # ADD: sanity check — fail fast if anything is still on CPU
        for name, param in self.model.named_parameters():
            if param.device.type == "cpu":
                raise RuntimeError(
                    f"Parameter '{name}' is still on CPU after quantization. "
                    "Not enough VRAM — try fewer GPUs layers or a smaller model."
                )

        print("[DEBUG] Model + tokenizer ready", file=sys.stderr, flush=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("[DEBUG] Model + tokenizer ready", file=sys.stderr, flush=True)

    def generate(self, prompt: str, system_prompt: str = None, max_new_tokens: int = 512, temperature: float = 0.0) -> str:
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

    def extract_claims(self, intervention_text: str) -> Dict[str, Any]:
        user_prompt = build_claim_extraction_prompt(intervention_text)

        raw_output = self.generate(
            prompt=user_prompt,
            system_prompt=CLAIM_EXTRACTION_SYSTEM_PROMPT,
            max_new_tokens=512,
            temperature=0.0
        )

        print("[DEBUG] Raw generation received", file=sys.stderr, flush=True)

        parsed = extract_json_with_repair(raw_output, llm=self)
        validated = validate_claim_extraction_output(parsed)

        return {
            "raw_model_output": raw_output,
            "parsed_output": validated
        }

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

    # Attempt 2: light normalization
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

CLAIM_EXTRACTION_SYSTEM_PROMPT = """
Je bent een annotator van parlementaire tekst.

Taak:
extraheer alle minimale argumentatieve tekstfragmenten uit de huidige interventie.

Regels:
- Extraheer alleen tekstfragmenten uit de interventie die je krijgt.
- Herschrijf niets.
- Vat niets samen.
- Normaliseer niets.
- Geef alleen exacte tekstfragmenten terug.
- Extraheer ALLE afzonderlijke argumentatieve eenheden, niet slechts enkele voorbeelden.
- Splits lange zinnen op in meerdere fragmenten als ze meerdere argumenten of conclusies bevatten.
- Geef de voorkeur aan juridische, beleidsmatige en redenerende claims boven retorische inkleding of illustratieve voorbeelden.
- Neem expliciete conclusies met signaalwoorden zoals "dus", "daarmee", "bovendien", "precies hetzelfde", "er is geen verschil", "dat betekent" afzonderlijk op wanneer ze argumentatieve waarde hebben.
- Een fragment mag een volledige zin zijn, maar ook een deelzin of korte frase.
- Kies de kleinst mogelijke fragmenten die nog zelfstandig argumentatieve betekenis hebben.
- Negeer begroetingen, procedurele opmerkingen, grapjes en inhoudsloze herhaling.
- Geef uitsluitend geldige JSON terug.
- Gebruik alleen het veld "quote".
""".strip()


CLAIM_EXTRACTION_USER_PROMPT_TEMPLATE = """
Extraheer de minimale argumentatieve tekstfragmenten uit deze interventie.

Neem alleen fragmenten die:
- een standpunt uitdrukken,
- een reden geven,
- beleid beoordelen,
- een gevolg benoemen,
- of een positie aanvallen of verdedigen.

Negeer begroetingen, procedurele opmerkingen, grapjes en inhoudsloze herhaling.

Output:
{{
  "claims": [
    {{"quote": "exact tekstfragment uit de interventie"}}
  ]
}}

Als er geen argumentatieve fragmenten zijn:
{{"claims": []}}

Voorbeeld:
{examples}

Interventie:
\"\"\"
{text}
\"\"\"
""".strip()

CLAIM_EXTRACTION_EXAMPLES = """
Voorbeeld:

Interventie:
"Wij vinden dat deze maatregel noodzakelijk is, omdat hij de veiligheid vergroot. Bovendien is er geen minder ingrijpend alternatief beschikbaar, dus het is gerechtvaardigd om deze stap te nemen."

Output:
{{
  "claims": [
    {{"quote": "deze maatregel noodzakelijk is"}},
    {{"quote": "hij de veiligheid vergroot"}},
    {{"quote": "er geen minder ingrijpend alternatief beschikbaar is"}},
    {{"quote": "het is gerechtvaardigd om deze stap te nemen"}}
  ]
}}
""".strip()

def build_claim_extraction_prompt(text: str) -> str:
    return CLAIM_EXTRACTION_USER_PROMPT_TEMPLATE.format(
        text=text.strip(),
        examples=CLAIM_EXTRACTION_EXAMPLES
    )

def validate_claim_extraction_output(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {"claims": []}

    claims = data.get("claims", [])
    if not isinstance(claims, list):
        return {"claims": []}

    cleaned_claims = []

    for item in claims:
        if not isinstance(item, dict):
            continue

        quote = item.get("quote", "")

        if not isinstance(quote, str) or not quote.strip():
            continue

        cleaned_claims.append({
            "quote": quote.strip()
        })

    return {"claims": cleaned_claims}