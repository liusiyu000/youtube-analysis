import re

import pandas as pd

STYLE_PATTERNS = {
    "comedy":          [r"\bcomedy\b", r"\bfunny\b", r"\bsketch\b", r"\bskit\b", r"\bparody\b", r"\bspoof\b", r"\bstand[- ]?up\b"],
    "challenge":       [r"\bchallenge\b", r"\b24[- ]?hour(s)?\b", r"\blast\s*(one|to)\b", r"\bextreme\b.*\bchallenge\b"],
    "reaction":        [r"\breaction\b", r"\breacts?\b", r"\bfirst\s*time\s*hearing\b"],
    "prank":           [r"\bprank\b", r"\bpranking\b", r"\bhidden\s*camera\b", r"\bpublic\s*prank\b"],
    "gaming":          [r"\bgaming\b", r"\bgamer(s)?\b", r"\bgameplay\b", r"\blet'?s?\s*play\b", r"\bspeedrun\b", r"\bminecraft\b", r"\bfortnite\b"],
    "vlog":            [r"\bvlog\b", r"\bdaily\s*vlog\b", r"\bday\s*in\s*the\s*life\b", r"\btravel\s*vlog\b", r"\bvlogmas\b"],
    "unboxing":        [r"\bunbox(?:ing)?\b", r"\bhaul\b", r"\bin\s*the\s*box\b", r"\bfirst\s*look\b", r"\btry[- ]?on\b"],
    "howto":           [r"\bhow[- ]?to\b", r"\btutorial\b", r"\bguide\b", r"\bdiy\b", r"\bstep[- ]?by[- ]?step\b", r"\brecipe\b", r"\bmake\b.*\b(?:yourself|at\s*home)\b"],
    "educational":     [r"\beducational\b", r"\bcrash\s*course\b", r"\bexplained?\b", r"\blesson\b", r"\bdocumentary\b", r"\bscience\b", r"\bhistory\b"],
    "product_review":  [r"\breview\b", r"\bhands[- ]?on\b", r"\bfirst\s*impression(s)?\b", r"\bpros\s*and\s*cons\b", r"\bbenchmark\b", r"\bcomparison\b", r"\bvs\.?\b"],
    "commentary":      [r"\bcommentary\b", r"\bopinion\b", r"\bmy\s*thoughts\b", r"\banalysis\b", r"\brant\b", r"\bresponse\b"],
    "video_essay":     [r"\bvideo\s*essay\b", r"\bdeep\s*dive\b", r"\bin[- ]?depth\b", r"\bretrospective\b"],
    "asmr":            [r"\basmr\b", r"\btingle(s)?\b", r"\bwhisper(ing)?\b", r"\bsoft\s*spoken\b"],
    "qna":             [r"\bq\s?&\s?a\b", r"\bask\s*me\s*anything\b", r"\banswering\s*your\s*questions\b", r"\bama\b"],

    "music":           [r"\bmusic\b", r"\bofficial\s*audio\b", r"\blyrics?\b", r"\bcover\s*song\b", r"\bofficial\s*mv\b", r"\bkaraoke\b"],
    "dance":           [r"\bdance\b", r"\bdance\s*cover\b", r"\bchoreography\b", r"\b舞蹈\b"],
    "sports":          [r"\bsports?\b", r"\bhighlights?\b", r"\bmatch\b", r"\bgame\s*recap\b", r"\blive\s*score\b", r"\bnba\b", r"\bfootball\b"],
    "news":            [r"\bnews\b", r"\bheadlines\b", r"\bbreaking\b\s*news\b", r"\bupdate(s)?\b", r"\bdaily\s*briefing\b"],
    "live_stream":     [r"\blive\s*stream(ed|ing)?\b", r"\bstream\s*replay\b", r"\blive\s*chat\b"],
    "podcast":         [r"\bpodcast\b", r"\baudio\s*only\b", r"\bepisod( e|es)\b", r"\binterview\b", r"\bsit[- ]?down\b"],
    "animation":       [r"\banimation\b", r"\banimated\b", r"\bcartoon\b", r"\banime\b", r"\b3d\s*short\b"],
    "documentary":     [r"\bdocumentary\b", r"\bdocu[- ]?series\b", r"\btrue\s*story\b", r"\bbehind\s*the\s*scenes\b"],
    "fitness":         [r"\bworkout\b", r"\bfitness\b", r"\bexercise\b", r"\btraining\b", r"\bab\s*routine\b", r"\byoga\b"],
    "beauty_fashion":  [r"\bmake[- ]?up\b", r"\bmakeup\b", r"\bbeauty\b", r"\bskincare\b", r"\bfashion\b", r"\boufit\b", r"\bgrwm\b"],
    "food_cooking":    [r"\bcooking\b", r"\brecipe\b", r"\bfood\b", r"\bmeal\s*prep\b", r"\bchef\b", r"\bwhat\s*i\s*eat\b"],
    "finance_business":[r"\bfinance\b", r"\bstock(s)?\b", r"\binvesting\b", r"\btrading\b", r"\bcrypto\b", r"\bbusiness\b"],
    "motivation":      [r"\bmotivation\b", r"\binspiration\b", r"\bself[- ]?improve(ment)?\b", r"\bmindset\b", r"\bproductivity\b"],
    "pets_animals":    [r"\bpet\b", r"\banimal\b", r"\bdog\b", r"\bcat\b", r"\bpup(py)?\b", r"\bkittens?\b"],
    "travel":          [r"\btravel\b", r"\btravel\s*vlog\b", r"\bexploring\b", r"\bbackpacking\b", r"\btrip\b", r"\badventure\b"],
    "shorts":          [r"\b#?shorts\b", r"\bvertical\b", r"\b15\s*sec(ond)?\b", r"\byoutube\s*shorts?\b"],

}
STYLE_LABELS = list(STYLE_PATTERNS.keys())
TEXT_FIELDS   = ["title", "description", "tags"]

# ---------- 2. 核心函数 ----------
def _extract_styles(row: pd.Series) -> list:
    text = " ".join(str(row[f]).lower() for f in TEXT_FIELDS if pd.notna(row[f]))
    hits = set()
    for label, patterns in STYLE_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, text):
                hits.add(label)
                break
    return list(hits)


def add_style_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["content_styles"] = df.apply(_extract_styles, axis=1)
    for lbl in STYLE_LABELS:
        df[f"style_{lbl}"] = df["content_styles"].apply(lambda x: int(lbl in x))
    return df
