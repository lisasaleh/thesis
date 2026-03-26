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
        token_args = {"token": hf_token} if hf_token else {}

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **token_args)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            **token_args,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt: str, max_new_tokens: int = 500, temperature: float = 0.0) -> str:
        messages = [{"role": "user", "content": prompt}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        model_inputs = self.tokenizer([text], return_tensors="pt")
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

    def extract_claims(self, summary: str, intervention_text: str) -> Dict[str, Any]:
        user_prompt = build_claim_extraction_prompt(summary, intervention_text)
        full_prompt = (
            f"{CLAIM_EXTRACTION_SYSTEM_PROMPT}\n\n"
            f"{user_prompt}"
        )

        raw_output = self.generate(
            prompt=full_prompt,
            max_new_tokens=700,
            temperature=0.0
        )

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
        raise ValueError(f"No JSON object found:\n{text}")

    candidate = match.group(0).strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    normalized = candidate
    normalized = re.sub(r'""', '"', normalized)
    normalized = re.sub(r",\s*([}\]])", r"\1", normalized)

    try:
        return json.loads(normalized)
    except json.JSONDecodeError:
        pass

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
            max_new_tokens=700,
            temperature=0.0,
        )
        repaired = _strip_fences(repaired)

        match_repaired = re.search(r"\{.*", repaired, flags=re.DOTALL)
        if match_repaired:
            repaired = match_repaired.group(0).strip()

        repaired = re.sub(r'""', '"', repaired)
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)

        try:
            return json.loads(repaired)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Repair failed.\nOriginal:\n{candidate}\n\nRepaired:\n{repaired}\n\nError: {e}"
            )

    raise ValueError(f"JSON parsing failed:\n{candidate}")


CLAIM_EXTRACTION_SYSTEM_PROMPT = """
Je bent een deskundige analist van parlementaire debatten en argumentatie.

Je taak is om minimale argumentatieve tekstfragmenten te extraheren uit ALLEEN de HUIDIGE interventie.

Regels:
- Gebruik de samenvatting alleen als context voor interpretatie.
- Extraheer fragmenten ALLEEN uit de huidige interventie.
- Kopieer NOOIT tekst uit de samenvatting.
- De samenvatting is ALLEEN bedoeld als context.
- Bewaar de oorspronkelijke formulering exact in het veld "quote".
- Verzin geen tekst.
- Geef de voorkeur aan korte, inhoudelijke argumentatieve eenheden boven lange passages.
- Een eenheid mag een volledige zin zijn, maar ook een deelzin of kort tekstfragment.
- Negeer begroetingen, procedurele opmerkingen, grapjes, tussenwerpsels en herhalingen zonder inhoudelijke argumentatieve waarde.
- Geef uitsluitend geldige JSON terug, zonder toelichting of extra tekst.
""".strip()


CLAIM_EXTRACTION_EXAMPLES = """
Voorbeeld 1
Huidige interventie:
"Wij steunen dit voorstel, maar het legt te veel druk op gemeenten."

Output:
{
  "claims": [
    {
      "quote": "Wij steunen dit voorstel",
      "type": "claim",
      "normalized": "Wij steunen dit voorstel.",
      "explanation": "Drukt een standpunt uit."
    },
    {
      "quote": "het legt te veel druk op gemeenten",
      "type": "reason",
      "normalized": "Dit voorstel legt te veel druk op gemeenten.",
      "explanation": "Geeft een onderbouwing."
    }
  ]
}

Voorbeeld 2
Huidige interventie:
"Dank u wel, voorzitter. Ik zal het kort houden."

Output:
{
  "claims": []
}
""".strip()


CLAIM_EXTRACTION_USER_PROMPT_TEMPLATE = """
Je krijgt:
1. Een samenvatting van het debat tot vlak vóór de huidige interventie.
2. De huidige interventie van de doelpartij.

Doel:
Extraheer alle minimale argumentatieve eenheden uit ALLEEN de huidige interventie die een van de volgende functies hebben:
- een politiek standpunt of claim,
- een reden of onderbouwing,
- een beoordeling van beleid,
- een genoemd gevolg of verwachte consequentie,
- een verdedigde of bestreden positie.

Definities:
- claim: een propositionele uitspraak, stelling, oordeel of positie die betwist of ondersteund kan worden.
- reason: een premisse, rechtvaardiging of onderbouwing voor of tegen een claim.

Outputformaat:
{{
  "claims": [
    {{
      "quote": "exact tekstfragment uit de huidige interventie",
      "type": "claim" of "reason",
      "normalized": "korte zelfstandige herformulering in het Nederlands",
      "explanation": "zeer korte uitleg"
    }}
  ]
}}

Vereisten:
- Als er geen argumentatieve eenheden aanwezig zijn, geef dan {{"claims": []}} terug.
- Extraheer uitsluitend uit de huidige interventie, nooit uit de samenvatting.
- "quote" moet exact overeenkomen met tekst uit de huidige interventie.
- "normalized" moet de quote herschrijven tot een korte, zelfstandige propositie.
- Houd "explanation" heel kort.
- Geef uitsluitend geldige JSON terug.

Voorbeelden:
{examples}

Samenvatting vóór de huidige interventie:
\"\"\"
{summary}
\"\"\"

Huidige interventie:
\"\"\"
{text}
\"\"\"
""".strip()


def build_claim_extraction_prompt(summary: str, text: str) -> str:
    return CLAIM_EXTRACTION_USER_PROMPT_TEMPLATE.format(
        examples=CLAIM_EXTRACTION_EXAMPLES,
        summary=summary.strip(),
        text=text.strip()
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
        claim_type = item.get("type", "claim")
        normalized = item.get("normalized", "")
        explanation = item.get("explanation", "")

        if not isinstance(quote, str) or not quote.strip():
            continue

        if claim_type not in {"claim", "reason"}:
            claim_type = "claim"

        if not isinstance(normalized, str):
            normalized = ""

        if not isinstance(explanation, str):
            explanation = ""

        cleaned_claims.append({
            "quote": quote.strip(),
            "type": claim_type,
            "normalized": normalized.strip(),
            "explanation": explanation.strip()
        })

    return {"claims": cleaned_claims}