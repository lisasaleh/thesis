from typing import Optional


def build_incremental_summary_prompt(
    current_state_json: Optional[str],
    speaker: str,
    party: str,
    idx: int,
    new_intervention_text: str,
    max_words: int = 250,
) -> str:
    current_state_block = current_state_json if current_state_json else "Nog geen samenvatting."

    return f"""
Je helpt bij het incrementeel bijhouden van een compacte, gestructureerde samenvatting van een Nederlands parlementair debat.

Taak:
Werk de lopende debatstaat bij op basis van de NIEUWE INTERVENTIE.

Doel:
De output moet een compacte representatie blijven van het HELE debat tot nu toe, niet alleen van de laatste interventie.

Belangrijke regels:
- Schrijf alle tekst volledig in het Nederlands.
- Gebruik alleen informatie uit de bestaande debatstaat en de nieuwe interventie.
- Verzin geen informatie.
- Behoud belangrijke eerdere context.
- Verwijder eerdere discussiepunten alleen als zij duidelijk niet langer relevant zijn.
- Als de nieuwe interventie voortbouwt op een bestaand punt, werk dat punt bij.
- Focus op inhoudelijke politieke inhoud.
- Negeer begroetingen, procedurele opmerkingen, humor en retorische opvulling, tenzij inhoudelijk relevant.
- Vat het debat samen als een kleine verzameling terugkerende discussiepunten.
- Gebruik maximaal 3 discussiepunten.
- Gebruik maximaal 2 argumenten per discussiepunt.
- Houd argumenten kort en kernachtig.
- Zorg dat "updated_summary" een compacte samenvatting is van het gehele debat tot nu toe.
- Houd "updated_summary" onder de {max_words} woorden.

JSON-regels:
- Geef EXACT één JSON-object terug.
- Gebruik alleen geldige JSON.
- Gebruik dubbele aanhalingstekens.
- Gebruik geen trailing commas.
- Geef geen tekst buiten het JSON-object.

JSON-schema:
{{
  "main_topic": "",
  "points_of_discussion": [
    {{
      "point": "",
      "arguments": [
        ""
      ]
    }}
  ],
  "updated_summary": ""
}}

Voorbeeld:
{{
  "main_topic": "Intrekking van het Nederlanderschap bij terroristische misdrijven",
  "points_of_discussion": [
    {{
      "point": "Proportionaliteit van de maatregel",
      "arguments": [
        "De minister stelt dat proportionaliteit gewaarborgd is.",
        "Critici vrezen dat de maatregel te ver gaat."
      ]
    }}
  ],
  "updated_summary": "Het debat gaat over de proportionaliteit en rechtsstatelijke legitimiteit van het intrekken van het Nederlanderschap bij terroristische misdrijven."
}}

HUIDIGE LOPENDE DEBATSTAAT:
{current_state_block}

METADATA VAN DE NIEUWE INTERVENTIE:
Spreker: {speaker}
Partij: {party}
Interventie-index: {idx}

NIEUWE INTERVENTIE:
\"\"\"
{new_intervention_text}
\"\"\"
""".strip()