import json
import re
from typing import Dict, Any


NORMALIZATION_SYSTEM_PROMPT = """
Je bent een annotator van parlementaire debatten.

Taak:
herformuleer één exact tekstfragment uit een parlementaire interventie tot één korte, volledig zelfstandige propositie.

Doel:
maak van het fragment een punt dat ook volledig begrijpelijk is zonder toegang tot de rest van het debat.

Regels:
- Gebruik het fragment als uitgangspunt.
- Gebruik de interventie, de samenvatting en de vorige interventies alleen om verwijzingen en context op te lossen.
- Verander de inhoudelijke betekenis niet.
- Voeg geen nieuwe claims, interpretaties of evaluaties toe.
- Maak alle vage verwijzingen expliciet als dat op basis van de context mogelijk is.
- Vervang voornaamwoorden, "dit", "dat", "die", "deze", "beide", "de wet", "het voorstel" en vergelijkbare verwijzingen door concrete omschrijvingen als de context dat toelaat.
- Het resultaat moet volledig begrijpelijk zijn zonder extra context.
- Vermijd formuleringen als "volgens de spreker", "hiermee", "in beide situaties", "dit voorstel" of "deze wet" als die niet volledig expliciet zijn.
- Houd het resultaat kort en zakelijk.
- Geef precies één zelfstandige propositie in het Nederlands terug.
- Geef uitsluitend geldige JSON terug.
- Gebruik alleen het veld "point".
""".strip()


NORMALIZATION_EXAMPLES = """
Voorbeeld 1

Quote:
"Er is dus wat de VVD-fractie betreft geen intrinsiek verschil tussen die twee."

Interventie:
"De wet geldt ook voor terroristen die in Nederland actief zijn. Er is dus wat de VVD-fractie betreft geen intrinsiek verschil tussen die twee."

Context:
Het debat gaat over het verschil tussen terroristen in Nederland en personen in buitenlandse krijgsdienst.

Output:
{
  "point": "Volgens de VVD is er geen intrinsiek verschil tussen terroristen in Nederland en personen in buitenlandse krijgsdienst."
}

Voorbeeld 2

Quote:
"Daarom moet de wet in beide gevallen worden toegepast."

Interventie:
"Er is geen principieel verschil tussen deze twee situaties. Daarom moet de wet in beide gevallen worden toegepast."

Context:
Het debat gaat over twee situaties die juridisch gelijk behandeld moeten worden.

Output:
{
  "point": "De wet moet volgens de spreker in beide situaties worden toegepast."
}
""".strip()


NORMALIZATION_USER_PROMPT_TEMPLATE = """
Herformuleer het onderstaande fragment tot één korte, zelfstandige propositie die buiten de oorspronkelijke context begrijpelijk is.

Belangrijk:
- Gebruik de context alleen om het fragment begrijpelijk te maken.
- Verander de inhoud niet.
- Voeg niets nieuws toe.
- Geef precies één punt terug.

Outputformaat:
{{
  "point": "korte zelfstandige propositie in het Nederlands"
}}

Voorbeelden:
{examples}

Samenvatting van het debat vóór deze interventie:
\"\"\"
{summary}
\"\"\"

Twee vorige interventies:
\"\"\"
{previous_interventions}
\"\"\"

Huidige interventie:
\"\"\"
{intervention}
\"\"\"

Te normaliseren quote:
\"\"\"
{quote}
\"\"\"
""".strip()


def build_normalization_prompt(
    quote: str,
    intervention: str,
    summary: str,
    previous_interventions: str,
) -> str:
    return NORMALIZATION_USER_PROMPT_TEMPLATE.format(
        examples=NORMALIZATION_EXAMPLES,
        quote=quote.strip(),
        intervention=intervention.strip(),
        summary=summary.strip(),
        previous_interventions=previous_interventions.strip(),
    )


def strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def validate_normalization_output(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {"point": ""}

    point = data.get("point", "")
    if not isinstance(point, str):
        point = ""

    return {"point": point.strip()}


def extract_json_with_basic_repair(text: str) -> Dict[str, Any]:
    text = strip_fences(text)

    match = re.search(r"\{.*", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Geen JSON-object gevonden:\n{text}")

    candidate = match.group(0).strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parsing mislukt:\n{candidate}\n\nError: {e}")