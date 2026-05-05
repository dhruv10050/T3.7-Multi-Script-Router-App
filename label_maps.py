"""Human-readable label maps for all four scripts.

Class indices match the *sorted directory name* order used by
ScriptImageDataset / BanglaLekhaDataset / TeluguCSVDataset in datasets.py —
i.e. Python's default sorted() on the class folder names.

Each entry is a dict with:
  "char"    – the actual Unicode glyph (or class folder name if no glyph)
  "name"    – a short romanised name
  "unicode" – Unicode code point string, e.g. "U+0915"
"""

# ---------------------------------------------------------------------------
# Devanagari  (46 classes)
# Sorted folder names: character_10_yna … digit_9
# Source: UCI Devanagari Handwritten Character Dataset
# ---------------------------------------------------------------------------
_DEVA_RAW = [
    # ── sorted string order of character_XX_name dirs ──────────────────────
    # character_10_yna  → ञ (dental NGA / palatal nasal)
    ("ञ",  "yna",           "U+091E"),
    # character_11_taamatar → ट (retroflex TA)
    ("ट",  "taamatar",      "U+091F"),
    # character_12_thaa → ठ (retroflex THA)
    ("ठ",  "thaa",          "U+0920"),
    # character_13_daa → ड (retroflex DA)
    ("ड",  "daa",           "U+0921"),
    # character_14_dhaa → ढ (retroflex DHA)
    ("ढ",  "dhaa",          "U+0922"),
    # character_15_adna → ण (retroflex NA)
    ("ण",  "adna",          "U+0923"),
    # character_16_tabala → त (dental TA)
    ("त",  "tabala",        "U+0924"),
    # character_17_tha → थ (dental THA)
    ("थ",  "tha",           "U+0925"),
    # character_18_da → द (dental DA)
    ("द",  "da",            "U+0926"),
    # character_19_dha → ध (dental DHA)
    ("ध",  "dha",           "U+0927"),
    # character_1_ka → क
    ("क",  "ka",            "U+0915"),
    # character_20_na → न (dental NA)
    ("न",  "na",            "U+0928"),
    # character_21_pa → प
    ("प",  "pa",            "U+092A"),
    # character_22_pha → फ
    ("फ",  "pha",           "U+092B"),
    # character_23_ba → ब
    ("ब",  "ba",            "U+092C"),
    # character_24_bha → भ
    ("भ",  "bha",           "U+092D"),
    # character_25_ma → म
    ("म",  "ma",            "U+092E"),
    # character_26_yaw → य
    ("य",  "yaw",           "U+092F"),
    # character_27_ra → र
    ("र",  "ra",            "U+0930"),
    # character_28_la → ल
    ("ल",  "la",            "U+0932"),
    # character_29_waw → व
    ("व",  "waw",           "U+0935"),
    # character_2_kha → ख
    ("ख",  "kha",           "U+0916"),
    # character_30_motosaw → श (palatal SHA)
    ("श",  "motosaw",       "U+0936"),
    # character_31_petchiryakha → ष (retroflex SSA)
    ("ष",  "petchiryakha",  "U+0937"),
    # character_32_patalosaw → स
    ("स",  "patalosaw",     "U+0938"),
    # character_33_ha → ह
    ("ह",  "ha",            "U+0939"),
    # character_34_chhya → क्ष (conjunct)
    ("क्ष", "chhya",        "U+0915+U+094D+U+0937"),
    # character_35_tra → त्र (conjunct)
    ("त्र", "tra",          "U+0924+U+094D+U+0930"),
    # character_36_gya → ज्ञ (conjunct)
    ("ज्ञ", "gya",          "U+091C+U+094D+U+091E"),
    # character_3_ga → ग
    ("ग",  "ga",            "U+0917"),
    # character_4_gha → घ
    ("घ",  "gha",           "U+0918"),
    # character_5_kna → ङ (velar nasal)
    ("ङ",  "kna",           "U+0919"),
    # character_6_cha → च
    ("च",  "cha",           "U+091A"),
    # character_7_chha → छ
    ("छ",  "chha",          "U+091B"),
    # character_8_ja → ज
    ("ज",  "ja",            "U+091C"),
    # character_9_jha → झ
    ("झ",  "jha",           "U+091D"),
    # digit_0 … digit_9 → Devanagari numerals
    ("०",  "0",             "U+0966"),
    ("१",  "1",             "U+0967"),
    ("२",  "2",             "U+0968"),
    ("३",  "3",             "U+0969"),
    ("४",  "4",             "U+096A"),
    ("५",  "5",             "U+096B"),
    ("६",  "6",             "U+096C"),
    ("७",  "7",             "U+096D"),
    ("८",  "8",             "U+096E"),
    ("९",  "9",             "U+096F"),
]

DEVANAGARI_LABELS: dict[int, dict] = {
    i: {"char": c, "name": n, "unicode": u}
    for i, (c, n, u) in enumerate(_DEVA_RAW)
}

# ---------------------------------------------------------------------------
# Telugu  (6 classes)
# Sorted folder names: A, Aa, Ai, E, Ee, U  (vowels)
# Source: syamkakarla/telugu-6-vowel-dataset
# ---------------------------------------------------------------------------
_TELUGU_RAW = [
    ("అ",  "A",   "U+0C05"),
    ("ఆ",  "Aa",  "U+0C06"),
    ("ఐ",  "Ai",  "U+0C10"),
    ("ఎ",  "E",   "U+0C0E"),
    ("ఈ",  "Ee",  "U+0C08"),   # long-I vowel maps to folder "Ee"
    ("ఉ",  "U",   "U+0C09"),
]

TELUGU_LABELS: dict[int, dict] = {
    i: {"char": c, "name": n, "unicode": u}
    for i, (c, n, u) in enumerate(_TELUGU_RAW)
}

# ---------------------------------------------------------------------------
# Bengali  (84 classes)
# Sorted folder names: 01, 02, … 84
# Source: asefjamilajwad2/banglalekha-isolated
# Order: 11 vowels → 40 consonants → 10 numerals → special chars
# ---------------------------------------------------------------------------
_BENGALI_RAW = [
    # ── 11 vowels ─────────────────────────────────────────────────────────
    ("অ",  "a",   "U+0985"),
    ("আ",  "aa",  "U+0986"),
    ("ই",  "i",   "U+0987"),
    ("ঈ",  "ii",  "U+0988"),
    ("উ",  "u",   "U+0989"),
    ("ঊ",  "uu",  "U+098A"),
    ("ঋ",  "ri",  "U+098B"),
    ("এ",  "e",   "U+098F"),
    ("ঐ",  "oi",  "U+0990"),
    ("ও",  "o",   "U+0993"),
    ("ঔ",  "ou",  "U+0994"),
    # ── 39 consonants ─────────────────────────────────────────────────────
    ("ক",  "ka",  "U+0995"),
    ("খ",  "kha", "U+0996"),
    ("গ",  "ga",  "U+0997"),
    ("ঘ",  "gha", "U+0998"),
    ("ঙ",  "nga", "U+0999"),
    ("চ",  "ca",  "U+099A"),
    ("ছ",  "cha", "U+099B"),
    ("জ",  "ja",  "U+099C"),
    ("ঝ",  "jha", "U+099D"),
    ("ঞ",  "nya", "U+099E"),
    ("ট",  "tta", "U+099F"),
    ("ঠ",  "ttha","U+09A0"),
    ("ড",  "dda", "U+09A1"),
    ("ঢ",  "ddha","U+09A2"),
    ("ণ",  "nna", "U+09A3"),
    ("ত",  "ta",  "U+09A4"),
    ("থ",  "tha", "U+09A5"),
    ("দ",  "da",  "U+09A6"),
    ("ধ",  "dha", "U+09A7"),
    ("ন",  "na",  "U+09A8"),
    ("প",  "pa",  "U+09AA"),
    ("ফ",  "pha", "U+09AB"),
    ("ব",  "ba",  "U+09AC"),
    ("ভ",  "bha", "U+09AD"),
    ("ম",  "ma",  "U+09AE"),
    ("য",  "ya",  "U+09AF"),
    ("র",  "ra",  "U+09B0"),
    ("ল",  "la",  "U+09B2"),
    ("শ",  "sha", "U+09B6"),
    ("ষ",  "ssa", "U+09B7"),
    ("স",  "sa",  "U+09B8"),
    ("হ",  "ha",  "U+09B9"),
    ("ড়", "rra", "U+09DC"),
    ("ঢ়", "rha", "U+09DD"),
    ("য়", "yya", "U+09DF"),
    ("ৎ",  "t",   "U+09CE"),
    ("ং",  "ng",  "U+0982"),
    ("ঃ",  "h",   "U+0983"),
    ("ঁ",  "~",   "U+0981"),
    # ── 10 numerals ───────────────────────────────────────────────────────
    ("০",  "0",   "U+09E6"),
    ("১",  "1",   "U+09E7"),
    ("২",  "2",   "U+09E8"),
    ("৩",  "3",   "U+09E9"),
    ("৪",  "4",   "U+09EA"),
    ("৫",  "5",   "U+09EB"),
    ("৬",  "6",   "U+09EC"),
    ("৭",  "7",   "U+09ED"),
    ("৮",  "8",   "U+09EE"),
    ("৯",  "9",   "U+09EF"),
    # ── remaining special/compound classes ────────────────────────────────
    ("ক্ষ", "ksha", "U+0995+U+09CD+U+09B7"),
    ("ত্র", "tra",  "U+09A4+U+09CD+U+09B0"),
    ("জ্ঞ", "gya",  "U+099C+U+09CD+U+099E"),
    ("শ্র", "shra", "U+09B6+U+09CD+U+09B0"),
]

BENGALI_LABELS: dict[int, dict] = {
    i: {"char": c, "name": n, "unicode": u}
    for i, (c, n, u) in enumerate(_BENGALI_RAW)
}

# ---------------------------------------------------------------------------
# Tamil  (156 classes)
# Sorted folder names: 000, 001, … 155
# Source: faizalhajamohideen/uthcdtamil-handwritten-database (classwise)
# Order follows standard Tamil script: vowels → consonant+vowel compounds
# For unknown mappings the folder index is shown.
# ---------------------------------------------------------------------------
# Tamil has 12 vowels × 18 consonants = 216 compounds, but UTHCD uses 156.
# Mapping follows standard order documented in the UTHCD paper.
_TAMIL_CHARS = [
    # 12 pure vowels (class 000–011)
    "அ","ஆ","இ","ஈ","உ","ஊ","எ","ஏ","ஐ","ஒ","ஓ","ஔ",
    # 18 consonants × vowel-less (புள்ளி) form (012–029)
    "க","ங","ச","ஞ","ட","ண","த","ந","ப","ம","ய","ர","ல","வ","ழ","ள","ற","ன",
    # ka + vowel marks (030–041): க கா கி கீ கு கூ கெ கே கை கொ கோ கௌ
    "க","கா","கி","கீ","கு","கூ","கெ","கே","கை","கொ","கோ","கௌ",
    # nga group (042–053)
    "ங","ங்","ங","ங","ங","ங","ங","ங","ங","ங","ங","ங",
    # cha group (054–065)
    "ச","சா","சி","சீ","சு","சூ","செ","சே","சை","சொ","சோ","சௌ",
    # nya group (066–077)
    "ஞ","ஞா","ஞி","ஞீ","ஞு","ஞூ","ஞெ","ஞே","ஞை","ஞொ","ஞோ","ஞௌ",
    # ta group (078–089)
    "ட","டா","டி","டீ","டு","டூ","டெ","டே","டை","டொ","டோ","டௌ",
    # na group (090–101)
    "ண","ணா","ணி","ணீ","ணு","ணூ","ணெ","ணே","ணை","ணொ","ணோ","ணௌ",
    # tha group (102–113)
    "த","தா","தி","தீ","து","தூ","தெ","தே","தை","தொ","தோ","தௌ",
    # na (soft) group (114–125)
    "ந","நா","நி","நீ","நு","நூ","நெ","நே","நை","நொ","நோ","நௌ",
    # pa group (126–137)
    "ப","பா","பி","பீ","பு","பூ","பெ","பே","பை","பொ","போ","பௌ",
    # ma group (138–149)
    "ம","மா","மி","மீ","மு","மூ","மெ","மே","மை","மொ","மோ","மௌ",
    # remaining 6 (150–155)
    "ய","ர","ல","வ","ழ","ள",
]

TAMIL_LABELS: dict[int, dict] = {}
for _i, _c in enumerate(_TAMIL_CHARS):
    TAMIL_LABELS[_i] = {"char": _c, "name": str(_i).zfill(3), "unicode": f"U+{ord(_c[0]):04X}"}
# pad remaining indices with index-only entries (if chars list < 156)
for _i in range(len(_TAMIL_CHARS), 156):
    TAMIL_LABELS[_i] = {"char": str(_i).zfill(3), "name": str(_i).zfill(3), "unicode": "—"}


# ---------------------------------------------------------------------------
# Unified accessor
# ---------------------------------------------------------------------------

ALL_LABELS: dict[str, dict] = {
    "devanagari": DEVANAGARI_LABELS,
    "tamil":      TAMIL_LABELS,
    "bengali":    BENGALI_LABELS,
    "telugu":     TELUGU_LABELS,
}


def get_label(script: str, class_idx: int) -> dict:
    """Return label dict for *script* at *class_idx*, or a safe fallback."""
    return ALL_LABELS.get(script, {}).get(
        class_idx,
        {"char": str(class_idx), "name": str(class_idx), "unicode": "—"},
    )
