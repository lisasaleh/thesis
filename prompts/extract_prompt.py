from typing import Dict, Any

CLAIM_EXTRACTION_SYSTEM_PROMPT = """
Je bent een annotator van parlementaire tekst.

Taak:
Extraheer alle afzonderlijke argumentatieve tekstfragmenten uit de huidige interventie.

Regels:
- Extraheer alleen tekstfragmenten uit de interventie die je krijgt.
- Gebruik geen informatie uit samenvattingen, eerdere context of wereldkennis.
- Herschrijf niets.
- Vat niets samen.
- Normaliseer niets.
- Geef alleen exacte tekstfragmenten terug.
- Extraheer alle afzonderlijke argumentatieve eenheden, niet slechts enkele voorbeelden.
- Kies telkens het kleinste tekstfragment dat nog zelfstandig argumentatieve betekenis draagt.
- Splits lange zinnen op wanneer zij meerdere afzonderlijke argumentatieve eenheden bevatten.
- Geef de voorkeur aan inhoudelijke claims, redenen, conclusies, beleidsbeoordelingen en causale uitspraken.
- Neem expliciete conclusies met signaalwoorden zoals "dus", "daarmee", "bovendien", "dat betekent", "er is geen verschil" afzonderlijk op wanneer zij zelfstandige argumentatieve waarde hebben.
- Negeer begroetingen, procedurele opmerkingen, grapjes en inhoudsloze herhaling.
- Geef uitsluitend geldige JSON terug.
- Gebruik alleen het veld "quote".
""".strip()


CLAIM_EXTRACTION_USER_PROMPT_TEMPLATE = """
Extraheer de afzonderlijke argumentatieve tekstfragmenten uit deze interventie.

Neem alleen fragmenten die:
- een standpunt uitdrukken,
- een reden geven,
- beleid beoordelen,
- een gevolg benoemen,
- een vergelijking met argumentatieve functie maken,
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
Interventie:
"Wij vinden dat deze maatregel noodzakelijk is, omdat hij de veiligheid vergroot. Bovendien is er geen minder ingrijpend alternatief beschikbaar, dus het is gerechtvaardigd om deze stap te nemen."

Output:
{
  "claims": [
    {"quote": "deze maatregel noodzakelijk is"},
    {"quote": "omdat hij de veiligheid vergroot"},
    {"quote": "er geen minder ingrijpend alternatief beschikbaar is"},
    {"quote": "dus het is gerechtvaardigd om deze stap te nemen"}
  ]
}
""".strip()


def build_claim_extraction_prompt(text: str) -> str:
    return CLAIM_EXTRACTION_USER_PROMPT_TEMPLATE.format(
        text=text.strip(),
        examples=CLAIM_EXTRACTION_EXAMPLES,
    )