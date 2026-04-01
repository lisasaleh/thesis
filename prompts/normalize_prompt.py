import json
import re
from typing import Dict, Any


NORMALIZATION_SYSTEM_PROMPT = """
Je bent een annotator van parlementaire debatten.

Taak:
Herformuleer één exact tekstfragment uit een parlementaire interventie tot één korte, volledig zelfstandige propositie.

Doel:
Maak van het fragment een neutraal geformuleerd punt dat volledig begrijpelijk is zonder toegang tot de rest van het debat, terwijl de oorspronkelijke betekenis zo exact mogelijk behouden blijft.

Gebruik van context:
- Gebruik de huidige interventie, de samenvatting en de vorige interventies alleen om verwijzingen op te lossen.
- Gebruik context niet om nieuwe informatie, nieuwe actoren, nieuwe standpunten of extra interpretaties toe te voegen.
- Kies altijd de meest minimale herformulering die het fragment zelfstandig begrijpelijk maakt.

Zeer belangrijke regels:
- Gebruik het fragment als uitgangspunt.
- Verander de inhoudelijke betekenis niet.
- Behoud negatie, modaliteit, strekking en polariteit exact.
- Voeg geen nieuwe claims, interpretaties, evaluaties, causaliteit of details toe.
- Verwijder partijnamen, fractienamen, persoonsnamen en bronvermeldingen als zij niet strikt noodzakelijk zijn voor de inhoud.
- Vervang partijnamen, fractienamen of persoonsnamen NIET door "wij", "we", "de spreker", "de fractie", "volgens de spreker" of vergelijkbare bronvermeldingen.
- Formuleer het resultaat bij voorkeur als een neutrale, zelfstandige propositie.
- Behoud een actor alleen als die actor inhoudelijk onderdeel is van het punt en niet slechts de bron van het standpunt is.
- Maak vage verwijzingen expliciet als dat op basis van de context ondubbelzinnig mogelijk is.
- Als een verwijzing niet ondubbelzinnig kan worden opgelost, kies een neutrale, minimale formulering in plaats van te gokken.
- Als het fragment te onvolledig of te ambigu is om veilig te herformuleren zonder te gokken, geef dan een lege string terug als point.
- Herschrijf vragen als vraag als het citaat echt een vraag is.
- Herschrijf een retorische vraag alleen als stelling als de onderliggende strekking expliciet en ondubbelzinnig uit de context blijkt.
- Normaliseer slechts één punt.
- Houd het resultaat kort, grammaticaal en zakelijk.
- Geef precies één zelfstandige propositie in correct Nederlands terug.
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
  "point": "Er is geen intrinsiek verschil tussen terroristen in Nederland en personen in buitenlandse krijgsdienst."
}

Voorbeeld 2

Quote:
"Daarom moet de wet in beide gevallen worden toegepast."

Interventie:
"We hebben het over terroristen in Nederland en personen in buitenlandse krijgsdienst. Er is geen principieel verschil tussen deze twee situaties. Daarom moet de wet in beide gevallen worden toegepast."

Context:
Het debat gaat over terroristen in Nederland en personen in buitenlandse krijgsdienst.

Output:
{
  "point": "De wet moet worden toegepast op terroristen in Nederland en personen in buitenlandse krijgsdienst."
}


Voorbeeld 3

Quote:
"De VVD-fractie wil dat terroristen en soldaten van een vreemd leger gelijk worden behandeld."

Interventie:
"De VVD-fractie wil dat terroristen en soldaten van een vreemd leger gelijk worden behandeld."

Context:
Het debat gaat over gelijke behandeling van terroristen en personen in buitenlandse krijgsdienst.

Output:
{
  "point": "Terroristen en soldaten van een vreemd leger moeten gelijk worden behandeld."
}

Voorbeeld 4

Quote:
"Wij vinden dat onwenselijk."

Interventie:
"Wij vinden dat onwenselijk."

Context:
Het debat gaat over een uitzonderingsclausule in verdragen voor terroristen.

Output:
{
  "point": "Een uitzonderingsclausule in verdragen voor terroristen is onwenselijk."
}

Voorbeeld 5

Quote:
"dat we nemen"

Interventie:
"dat we nemen"

Context:
Onvoldoende context om ondubbelzinnig te bepalen waar het fragment naar verwijst.

Output:
{
  "point": ""
}
""".strip()


NORMALIZATION_USER_PROMPT_TEMPLATE = """
Herformuleer het onderstaande fragment tot één korte, zelfstandige propositie die buiten de oorspronkelijke context begrijpelijk is.

Belangrijk:
- Gebruik context alleen om verwijzingen op te lossen.
- Verander de inhoud niet.
- Voeg niets nieuws toe.
- Verwijder partijnamen, fractienamen en persoonsnamen als die niet noodzakelijk zijn voor de betekenis.
- Vervang zulke namen niet door "wij", "we", "de spreker" of "de fractie".
- Formuleer het resultaat als een neutrale zelfstandige propositie.
- Behoud negatie en strekking exact.
- Als veilige normalisatie niet mogelijk is zonder te gokken, geef dan een lege string terug.
- Geef precies één punt terug.

Outputformaat:
{
  "point": "korte zelfstandige propositie in het Nederlands"
}

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
        raise ValueError(f"Geen JSON-object gevonden:\\n{text}")

    candidate = match.group(0).strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parsing mislukt:\\n{candidate}\\n\\nError: {e}")