from typing import Dict, Any

CLAIM_EXTRACTION_SYSTEM_PROMPT = """
Je bent een annotator van parlementaire tekst.

Taak:
Extraheer alle afzonderlijke substantieve argumentatieve tekstfragmenten uit de huidige interventie.

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
- Neem expliciete conclusies met signaalwoorden zoals "dus", "daarmee", "bovendien", "dat betekent" afzonderlijk op wanneer zij zelfstandige argumentatieve waarde hebben.
- SLUIT UIT: speech acts, ontkenningen, correcties en meta-commentaar (bijv. "dat heb ik niet gezegd", "dat wilde ik niet zeggen", "ik zei dat niet", referenties naar "wat je zojuist zei" of "wat de regering doet").
- Negeer begroetingen, procedurele opmerkingen, grapjes en inhoudsloze herhaling.
- Geef uitsluitend geldige JSON terug.
- Gebruik alleen het veld "quote".
""".strip()


CLAIM_EXTRACTION_USER_PROMPT_TEMPLATE = """
Extraheer de afzonderlijke substantieve argumentatieve tekstfragmenten uit deze interventie.

Neem alleen fragmenten die:
- een standpunt over beleid of maatregelen uitdrukken,
- een concrete reden geven voor waarom iets goed/slecht is,
- beleid beoordelen of evalueren,
- een causaal verband beschrijven,
- een gevolg of implicatie benoemen,
- een vergelijking met argumentatieve functie maken.

SLUIT UIT:
- Speech acts zonder inhoud: "ik heb niet gezegd", "dat wilde ik niet zeggen", "dat zei ik niet"
- Louter formele ontkenningen zonder onderbouwing
- Verwijzingen naar wat anderen zeiden zonder eigen argument
- Meta-opmerkingen over de discussie zelf
- Procedurele opmerkingen
- Lege herhaling

Inclusief (als onderdeel van een argument):
- Ontkenningen met reden: "Dit is niet waar, omdat..." of "X is onjuist omdat Y"
- Correcties die een substantief argument maken

Output:
{{{{
  "claims": [
    {{"quote": "exact tekstfragment uit de interventie"}}
  ]
}}}}

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
{{
  "claims": [
    {{"quote": "deze maatregel noodzakelijk is"}},
    {{"quote": "omdat hij de veiligheid vergroot"}},
    {{"quote": "er geen minder ingrijpend alternatief beschikbaar is"}},
    {{"quote": "dus het is gerechtvaardigd om deze stap te nemen"}}
  ]
}}
""".strip()


def build_claim_extraction_prompt(text: str) -> str:
    return CLAIM_EXTRACTION_USER_PROMPT_TEMPLATE.format(
        text=text.strip(),
        examples=CLAIM_EXTRACTION_EXAMPLES,
    )