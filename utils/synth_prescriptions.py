# utils/synth_prescriptions.py
import random, re, json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from faker import Faker
import numpy as np

fake = Faker()

DRUGS = {
    "Metformin": ["250 mg", "500 mg", "850 mg", "1000 mg"],
    "Atorvastatin": ["10 mg", "20 mg", "40 mg", "80 mg"],
    "Lisinopril": ["5 mg", "10 mg", "20 mg", "40 mg"],
    "Amlodipine": ["2.5 mg", "5 mg", "10 mg"],
    "Omeprazole": ["10 mg", "20 mg", "40 mg"],
    "Levothyroxine": ["25 mcg", "50 mcg", "75 mcg", "100 mcg"],
    "Metoprolol": ["25 mg", "50 mg", "100 mg"],
    "Insulin Glargine": ["10 units", "20 units", "30 units"]
}

ROUTES = ["oral", "PO", "IV", "IM", "SC", "topical", "inh"]
FREQS  = ["qd", "qod", "bid", "tid", "qid", "q12h", "q8h", "q6h", "prn", "once daily", "twice daily"]
DURATIONS = ["for 5 days", "for 7 days", "for 10 days", "for 2 weeks", "for 1 month", None]
INSTRUX = ["with meals", "before breakfast", "at bedtime", "with water", "avoid driving", None]

SIG_TEMPLATES = [
    "{DRUG} {STRENGTH} {ROUTE} {FREQ} {DURATION} {INSTRUX}",
    "{DRUG} {STRENGTH} {FREQ} {ROUTE}",
    "{DRUG} {STRENGTH} {FREQ}",
    "{DRUG}: {STRENGTH}, {FREQ}, {ROUTE} {DURATION}",
]

ABBREV_MAP = {"oral":"PO","once daily":"qd","twice daily":"bid","three times daily":"tid","four times daily":"qid"}

def _maybe_abbrev(s: str, p: float) -> str:
    if random.random() > p: return s
    out = s
    for k,v in ABBREV_MAP.items():
        out = re.sub(rf"\b{k}\b", v, out, flags=re.I)
    return out

def _maybe_typos(s: str, p: float) -> str:
    if random.random() > p: return s
    arr = list(s); k = max(1, len(arr)//30)
    for _ in range(k):
        i = random.randrange(len(arr))
        if random.random() < 0.5:
            arr[i] = ""
        else:
            j = min(len(arr)-1, i+1); arr[i], arr[j] = arr[j], arr[i]
    return "".join(arr)

def _maybe_ocr_noise(s: str, p: float) -> str:
    if random.random() > p: return s
    swaps = {"0":"O","O":"0","1":"l","l":"1","5":"S","S":"5","mg":"m9","mcg":"mc9"}
    out = []
    for t in s.split():
        if random.random() < 0.15:
            for a,b in swaps.items():
                if a in t and random.random()<0.7:
                    t = t.replace(a,b)
        out.append(t)
    return " ".join(out)

@dataclass
class MedLine:
    drug: str; strength: str; route: str; freq: str; duration: str|None; instructions: str|None

def _make_med() -> MedLine:
    d = random.choice(list(DRUGS.keys()))
    return MedLine(
        d, random.choice(DRUGS[d]),
        random.choice(ROUTES), random.choice(FREQS),
        random.choice(DURATIONS), random.choice(INSTRUX)
    )

def _med_to_sig(m: MedLine) -> str:
    tpl = random.choice(SIG_TEMPLATES)
    s = tpl.format(DRUG=m.drug, STRENGTH=m.strength, ROUTE=m.route, FREQ=m.freq,
                   DURATION=(m.duration or ""), INSTRUX=(m.instructions or ""))
    s = re.sub(r"\s{2,}", " ", s).strip()
    s = _maybe_abbrev(s, p=0.6)
    if random.random() < 0.25: s = s.title()
    return s

def _layout_block(patient: Dict[str,Any], prescriber: Dict[str,Any], sigs: List[str]) -> str:
    header = f"""Patient: {patient['name']}    DOB: {patient['dob']}
Prescriber: Dr. {prescriber['name']}   NPI: {prescriber['npi']}
Rx Date: {patient['date']}
----------------------------------------
"""
    body = "\n".join(f"{i+1}. {s}" for i,s in enumerate(sigs))
    if random.random() < 0.3: body = body.replace(". ", ".\n")
    footer = "\n\nNotes: {random.choice(['', 'Take with food', 'Monitor BP', ''])}\n"
    return header + body + footer

def _fake_identity() -> Dict[str,Any]:
    return {"name": fake.name(),
            "dob": fake.date_of_birth(minimum_age=18, maximum_age=90).isoformat(),
            "date": fake.date_between(start_date="-2y", end_date="today").isoformat()}

def _fake_prescriber() -> Dict[str,Any]:
    return {"name": fake.name(), "npi": str(fake.random_number(digits=10, fix_len=True))}

def generate_sample(n_meds=1, noise=None) -> Dict[str,Any]:
    noise = noise or {"typo":0.2,"ocr":0.2}
    patient, prescriber = _fake_identity(), _fake_prescriber()
    meds = [_make_med() for _ in range(n_meds)]
    sigs = [_med_to_sig(m) for m in meds]
    sigs_noisy = [_maybe_ocr_noise(_maybe_typos(s, noise["typo"]), noise["ocr"]) for s in sigs]
    text = _layout_block(patient, prescriber, sigs_noisy)
    gt = {"patient": patient, "prescriber": prescriber, "medications": [asdict(m) for m in meds], "synthetic": True}
    return {"text": text, "ground_truth": gt}

def generate_dataset(n=100, min_meds=1, max_meds=3, noise=None, seed: int|None=42):
    if seed is not None: random.seed(seed); np.random.seed(seed)
    noise = noise or {"typo":0.2,"ocr":0.2}
    return [generate_sample(n_meds=random.randint(min_meds, max_meds), noise=noise) for _ in range(n)]