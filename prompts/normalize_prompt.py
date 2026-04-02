import json
import re
from typing import Dict, Any


NORMALIZATION_SYSTEM_PROMPT = """
Je bent een expert-annotator van Nederlandse parlementaire debatten.

KERNTAAK: Zet een QUOTE om naar een ZELFSTANDIGE CLAIM — begrijpelijk zonder te weten wie spreekt of wat er eerder is gezegd.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOOFDDOEL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

De spreker/partij IS de bron van de claim, NIET het onderwerp.
Verwijder dus de spreker als subject, maar maak het ONDERWERP expliciet.

  SLECHT: "De PvdA vindt dat dit een slecht idee is"
  GOED:   "Het sluiten van regionale ziekenhuizen is een slecht idee"

  SLECHT: "Wij steunen dit voorstel"
  GOED:   "Het voorstel voor gratis kinderopvang verdient steun"

  SLECHT: "Dat moeten we voorkomen"
  GOED:   "Belastingontwijking door multinationals moet voorkomen worden"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WERKWIJZE (volg in volgorde)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STAP 1 — IDENTIFICEER HET ONDERWERP
  Wat gaat de quote over? Gebruik de interventie, vorige interventies en samenvatting.
  Gebruik NOOIT de sprekende partij als onderwerp.

STAP 2 — LOS VAGE VERWIJZINGEN OP
  "dit" / "dat" / "het" / "zo'n maatregel" → vervang door het concrete onderwerp.
  ENKEL als het ondubbelzinnig uit de context volgt. Twijfel je? → geef "" terug.

STAP 3 — BEWAAR BETEKENIS PRECIES
  Behoud: negatie, modaliteit, twijfel, vragen, perspectief.
  Voeg NIETS toe wat niet in de quote staat.

STAP 4 — SCHRIJF ÉÉN ZELFSTANDIGE ZIN
  De zin moet begrijpelijk zijn zonder context. Correct Nederlands.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HARDE REGELS (GEEN UITZONDERINGEN)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BEWAAR ALTIJD:
   • Negatie: "niet", "geen", "nooit", "nergens"
   • Modaliteit: "kan", "moet", "mag", "zal", "zou", "hoeft"
   • Twijfel: "misschien", "mogelijk", "zou kunnen"
   • Vragen: vragen blijven vragen, geen stellingen ervan maken

VERWIJDER ALTIJD:
   • De sprekende partij als grammaticaal subject
   • Frases als: "Wij vinden...", "De SP stelt...", "Ik ben van mening dat..."

MAAK ALTIJD EXPLICIET:
   • Vage verwijzingen → concreet onderwerp uit context

NOOIT:
   • Inhoud verzinnen die niet in de quote staat
   • "kan" vervangen door "zal" of "vindt plaats"
   • De spreker als bron toevoegen: "Mevrouw X stelt dat..."
   • Andere partijen als subject verwijderen (die zijn het onderwerp, niet de spreker)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SPECIALE GEVALLEN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ANDERE PARTIJEN ALS ONDERWERP → neutraliseren:
  Quote: "D66 wil dit afschaffen"  (spreker is VVD)
  →  "Anderen willen de dividendbelasting afschaffen"
  →  of: "Een andere partij wil de dividendbelasting afschaffen"
  (D66 = andere partij dan spreker → vervang door neutraal label)

SPREKENDE PARTIJ ALS SUBJECT → verwijder, maak onderwerp expliciet:
  Quote: "Wij willen dit afschaffen"  (spreker is VVD)
  →  "De dividendbelasting moet afgeschaft worden"
  (Geen partijnaam, geen "wij", onderwerp wordt subject van de zin)

NOOIT een partijnaam in de output — ook niet als het een andere partij is.
Gebruik neutrale labels: "anderen", "een andere partij", "de coalitie", "de oppositie",
"de indieners", "de minister" (voor bewindspersonen), afhankelijk van context.

ONTKENNING VAN EIGEN UITSPRAAK → letterlijk bewaren:
  Quote: "Ik heb dat nooit gezegd"  →  "Ik heb dat nooit gezegd."
  (Geen interpretatie, negatie blijft staan)

TE VAAG ZONDER CONTEXT → lege string:
  Quote: "Dat zou verstandiger zijn"  +  onvoldoende context  →  ""
""".strip()

NORMALIZATION_EXAMPLES = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CORRECTE VOORBEELDEN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VOORBEELD 1 — Vaag onderwerp oplossen via context (jeugdzorg)
  Quote:        "Dat is een uitermate slecht idee"
  Interventie:  Mevrouw Westerveld (GroenLinks): Dat is een uitermate slecht idee.
  Context:      Debat over het opheffen van algemene jeugdzorginstellingen
  Output:
  {
    "point": "Het opheffen van algemene jeugdzorginstellingen is een uitermate slecht idee."
  }
  ✓ "dat" vervangen door concreet onderwerp; oordeel behouden

VOORBEELD 2 — Modaliteit behouden (stikstof)
  Quote:        "dit kan niet zomaar ingevoerd worden"
  Interventie:  De heer Grinwis (ChristenUnie): De stikstofwet kan niet zomaar ingevoerd worden.
  Output:
  {
    "point": "De stikstofwet kan niet zomaar ingevoerd worden."
  }
  ✓ "kan niet" behouden; geen versterking naar "mag niet" of "wordt niet"

VOORBEELD 3 — Andere partij neutraliseren (woningbouw)
  Quote:        "de VVD wil dit afschaffen"
  Interventie:  De heer Nijboer (PvdA): De VVD wil de huurtoeslag afschaffen.
  Output:
  {
    "point": "Een andere partij wil de huurtoeslag afschaffen."
  }
  ✓ VVD vervangen door neutraal label; "dit" opgelost via context

VOORBEELD 4 — Spreker verwijderen, onderwerp expliciet maken (defensie)
  Quote:        "Wij steunen dit volledig"
  Interventie:  De heer Brekelmans (VVD): Wij steunen dit volledig.
  Context:      Debat over verhoging van het defensiebudget naar 2% bbp
  Output:
  {
    "point": "De verhoging van het defensiebudget naar 2% bbp verdient volledige steun."
  }
  ✓ "Wij" verwijderd; "dit" opgelost via context

VOORBEELD 5 — Letterlijke ontkenning bewaren (eigen uitspraak)
  Quote:        "Ik heb dat niet gezegd"
  Interventie:  Mevrouw Agema (PVV): Ik heb dat niet gezegd.
  Output:
  {
    "point": "Ik heb dat niet gezegd."
  }
  ✓ Geen interpretatie; negatie letterlijk bewaard

VOORBEELD 6 — Vraag blijft vraag (toeslagenaffaire)
  Quote:        "wanneer komt er dan eindelijk duidelijkheid voor de gedupeerde ouders?"
  Interventie:  Mevrouw Marijnissen (SP): Wanneer komt er dan eindelijk duidelijkheid voor de gedupeerde ouders?
  Output:
  {
    "point": "Wanneer komt er eindelijk duidelijkheid voor de gedupeerde ouders in de toeslagenaffaire?"
  }
  ✓ Vraagvorm behouden; context gebruikt om "dan" weg te werken

VOORBEELD 7 — Te vaag, onvoldoende context
  Quote:        "dat moeten we echt aanpakken"
  Interventie:  De heer Paternotte (D66): Dat moeten we echt aanpakken.
  Context:      Geen ondubbelzinnige referent beschikbaar
  Output:
  {
    "point": ""
  }
  ✓ Gokken vermeden; lege string correct

VOORBEELD 8 — Twijfel behouden (klimaat)
  Quote:        "dat zou misschien kunnen helpen"
  Interventie:  Mevrouw Kröger (GroenLinks): Een CO2-heffing zou misschien kunnen helpen.
  Output:
  {
    "point": "Een CO2-heffing zou misschien kunnen helpen."
  }
  ✓ "zou misschien kunnen" volledig behouden; geen versterking

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FOUTE VOORBEELDEN (NOOIT DOEN)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FOUT A — Spreker toevoegen als bron
  Quote:    "Dit is onacceptabel"
  Fout:     "De heer Omtzigt vindt de gang van zaken onacceptabel."
  Correct:  "De gang van zaken rondom de toeslagenaffaire is onacceptabel."

FOUT B — Modaliteit veranderen
  Quote:    "dit kan fout gaan"
  Fout:     "Dit gaat fout."
  Correct:  "De uitvoering van de wet kan fout gaan."

FOUT C — Inhoud verzinnen
  Quote:    "Ik heb dat niet gezegd"
  Fout:     "De VVD ontkent betrokkenheid bij de beslissing."
  Correct:  "Ik heb dat niet gezegd."

FOUT D — Negatie verwijderen
  Quote:    "dit mag niet gebeuren"
  Fout:     "Dit gebeurt."
  Correct:  "Discriminatie op de arbeidsmarkt mag niet gebeuren."

FOUT E — Partijnaam in output laten staan
  Quote:    "de SGP wil dit terugdraaien"
  Fout:     "De SGP wil de euthanasiewet terugdraaien."
  Correct:  "Een andere partij wil de euthanasiewet terugdraaien."
""".strip()

NORMALIZATION_USER_PROMPT_TEMPLATE = """
TAAK: Zet de quote om naar ÉÉN zelfstandige, context-onafhankelijke zin.

KERNVRAAG: Als iemand deze zin leest zonder het debat te kennen — begrijpt die dan waarover het gaat?

CHECKLIST:
□ Is het onderwerp concreet (geen "dit", "dat", "het")?
□ Is de sprekende partij verwijderd als subject?
□ Zijn alle partijnamen vervangen (spreker → weggelaten, anderen → neutraal label)?
□ Zijn negatie en modaliteit ("niet", "kan", "moet", "zou") intact?
□ Heb ik niets toegevoegd wat niet in de quote stond?
□ Is de zin grammaticaal correct Nederlands?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Samenvatting van het debat tot nu toe:
\"\"\"
{summary}
\"\"\"

Vorige twee interventies:
\"\"\"
{previous_interventions}
\"\"\"

Huidige interventie (bron van de quote):
\"\"\"
{intervention}
\"\"\"

Te normaliseren quote:
\"\"\"
{quote}
\"\"\"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT (GELDIG JSON, NIETS ANDERS)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
  "point": "één zelfstandige zin, of lege string als te vaag"
}

Voorbeelden:
{examples}
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

    # Fix double braces {{ and }} that appear at start/end
    candidate = re.sub(r"^\{\{", "{", candidate)
    candidate = re.sub(r"\}\}$", "}", candidate)

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Remove trailing commas before closing braces
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parsing mislukt:\\n{candidate}\\n\\nError: {e}")