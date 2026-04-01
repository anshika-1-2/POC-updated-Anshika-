"""
storage.py
Three structured CSV stores:

data/users.csv             — registered users (one row per user)
data/scan_records.csv      — one row per scan  (linked by user_id + scan_id)
data/purchase_behavior.csv — one row per product-decision event
"""

import csv
import uuid
import hashlib
import shutil
import fcntl
from datetime import datetime
from pathlib import Path

DATA_DIR   = Path(__file__).parent.parent / "data"
IMAGES_DIR = DATA_DIR / "images"

USERS_FILE    = DATA_DIR / "users.csv"
RECORDS_FILE  = DATA_DIR / "scan_records.csv"
BEHAVIOR_FILE = DATA_DIR / "purchase_behavior.csv"

# ── Schema ────────────────────────────────────────────────────────────────────
USER_FIELDS = [
    "user_id", "username", "password_hash", "name", "allergen",
    "sex", "age", "height_cm", "weight_kg", "activity", "pregnancy",
    "registered_at",
]

SCAN_FIELDS = [
    "scan_id", "user_id", "username", "user_allergen",
    "dri_sex", "dri_age", "dri_height_cm", "dri_weight_kg",
    "dri_activity", "dri_bmi", "dri_eer_kcal",
    "image_path", "product_name", "ingredients",
    "detected_allergens", "detected_additives",
    "health_score", "health_grade", "health_verdict",
    "timestamp",
]

BEHAVIOR_FIELDS = [
    "behavior_id", "scan_id", "user_id", "product_name",
    "will_buy",                   # yes / no
    "paired_product_selected",    # yes / no / na
    "paired_product_name",
    "paired_product_id",
    "timestamp",
]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _hash(plain: str) -> str:
    return hashlib.sha256(plain.encode()).hexdigest()


def _ensure(path: Path, fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()


def _append(path: Path, fields: list[str], row: dict):
    _ensure(path, fields)
    with open(path, "a", newline="", encoding="utf-8") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX)
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _read(path: Path, fields: list[str]) -> list[dict]:
    _ensure(path, fields)
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _rewrite(path: Path, fields: list[str], rows: list[dict]):
    _ensure(path, fields)
    with open(path, "w", newline="", encoding="utf-8") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX)
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# ══════════════════════════════════════════════════════════════════════════════
# USER AUTH
# ══════════════════════════════════════════════════════════════════════════════

def register_user(username, password, name, allergen="", sex="", age=0,
                  height_cm=0.0, weight_kg=0.0, activity="", pregnancy="None") -> dict:
    if not username.strip():
        return {"ok": False, "error": "Username cannot be blank."}
    if not password:
        return {"ok": False, "error": "Password cannot be blank."}
    users = _read(USERS_FILE, USER_FIELDS)
    if any(u["username"].lower() == username.strip().lower() for u in users):
        return {"ok": False, "error": "Username already taken."}
    user_id = uuid.uuid4().hex[:10].upper()
    row = {
        "user_id": user_id, "username": username.strip(),
        "password_hash": _hash(password), "name": name.strip(),
        "allergen": allergen.strip().lower(), "sex": sex, "age": age,
        "height_cm": height_cm, "weight_kg": weight_kg,
        "activity": activity, "pregnancy": pregnancy,
        "registered_at": datetime.now().isoformat(timespec="seconds"),
    }
    _append(USERS_FILE, USER_FIELDS, row)
    return {"ok": True, "user_id": user_id, "user": row}


def login_user(username, password) -> dict:
    ph = _hash(password)
    for u in _read(USERS_FILE, USER_FIELDS):
        if u["username"].lower() == username.strip().lower() and u["password_hash"] == ph:
            return {"ok": True, "user": u}
    return {"ok": False, "error": "Invalid username or password."}


def update_user_profile(user_id: str, updates: dict) -> bool:
    users = _read(USERS_FILE, USER_FIELDS)
    found = False
    for u in users:
        if u["user_id"] == user_id:
            u.update(updates)
            found = True
            break
    if not found:
        return False
    _rewrite(USERS_FILE, USER_FIELDS, users)
    return True


def get_user_by_id(user_id: str) -> dict | None:
    for u in _read(USERS_FILE, USER_FIELDS):
        if u["user_id"] == user_id:
            return u
    return None


# ══════════════════════════════════════════════════════════════════════════════
# IMAGES
# ══════════════════════════════════════════════════════════════════════════════

def save_image(source_path: str) -> str:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    ext  = Path(source_path).suffix or ".jpg"
    dest = IMAGES_DIR / f"{uuid.uuid4().hex}{ext}"
    shutil.copy2(source_path, dest)
    return str(dest)


# ══════════════════════════════════════════════════════════════════════════════
# SCAN RECORDS
# ══════════════════════════════════════════════════════════════════════════════

def save_record(user_id, username, user_allergen, image_path, product_name,
                ingredients, detected_allergens, detected_additives,
                health_score=0, health_grade="", health_verdict="",
                dri: dict | None = None) -> str:
    scan_id    = uuid.uuid4().hex[:12].upper()
    dri_inp    = (dri or {}).get("inputs") or {}
    row = {
        "scan_id":           scan_id,
        "user_id":           user_id,
        "username":          username,
        "user_allergen":     user_allergen or "",
        "dri_sex":           dri_inp.get("sex", ""),
        "dri_age":           dri_inp.get("age", ""),
        "dri_height_cm":     dri_inp.get("height_cm", ""),
        "dri_weight_kg":     dri_inp.get("weight_kg", ""),
        "dri_activity":      dri_inp.get("activity", ""),
        "dri_bmi":           (dri or {}).get("bmi", ""),
        "dri_eer_kcal":      (dri or {}).get("eer", ""),
        "image_path":        image_path,
        "product_name":      product_name,
        "ingredients":       ingredients,
        "detected_allergens": "; ".join(detected_allergens),
        "detected_additives": "; ".join(f"{a}:{n}" for a, n in detected_additives),
        "health_score":      health_score,
        "health_grade":      health_grade,
        "health_verdict":    health_verdict,
        "timestamp":         datetime.now().isoformat(timespec="seconds"),
    }
    _append(RECORDS_FILE, SCAN_FIELDS, row)
    return scan_id


# ══════════════════════════════════════════════════════════════════════════════
# PURCHASE BEHAVIOR
# ══════════════════════════════════════════════════════════════════════════════

def save_behavior(scan_id, user_id, product_name, will_buy: bool,
                paired_product_selected: bool | None = None,
                paired_product_name="", paired_product_id="") -> str:
    bid = uuid.uuid4().hex[:10].upper()
    paired_sel = ("yes" if paired_product_selected else "no") if will_buy else "na"
    row = {
        "behavior_id":             bid,
        "scan_id":                 scan_id,
        "user_id":                 user_id,
        "product_name":            product_name,
        "will_buy":                "yes" if will_buy else "no",
        "paired_product_selected": paired_sel,
        "paired_product_name":     paired_product_name,
        "paired_product_id":       str(paired_product_id),
        "timestamp":               datetime.now().isoformat(timespec="seconds"),
    }
    _append(BEHAVIOR_FILE, BEHAVIOR_FIELDS, row)
    return bid