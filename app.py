"""
app.py  —  Japanese Food Label Allergen Scanner
Wide layout · DRI Calculator · Persistent inputs · Smart Pairing · Polished UI

Run:  streamlit run app.py
"""

import json
import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from services.detection    import analyze_ingredients, JP_MANDATORY
from services.storage      import save_image, save_record
from services.dri          import calculate_dri
from services.diet         import classify_diet
from services.health_score import compute_health_score
from services.smart_pairing import smart_pair, parse_db_nutrition

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Food Label Allergen Scanner",
    page_icon="🍱",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { background:#f4f6fb; }
[data-testid="stSidebar"] .block-container { padding-top: 1rem; }

.chip {
  display:inline-block; background:#eef2f7; color:#2d3748;
  border-radius:6px; padding:2px 9px; margin:2px; font-size:12.5px;
}
.pill {
  display:inline-block; border-radius:20px;
  padding:3px 13px; margin:3px; font-size:13px; font-weight:500;
}
.pill-red    { background:#fde8e8; color:#c0392b; }
.pill-yellow { background:#fef9e2; color:#9a6600; }
.pill-orange { background:#fef0e0; color:#c05000; }
.pill-green  { background:#e6f9ee; color:#1a6e3a; }
.pill-blue   { background:#e8f0fe; color:#1a56db; }

.sec-header {
  font-size:15px; font-weight:700; color:#1e293b;
  letter-spacing:.02em; margin:14px 0 6px 0;
}
[data-testid="stMetricValue"] { font-size:28px !important; font-weight:700 !important; }
[data-testid="stMetricLabel"] { font-size:12px !important; color:#64748b !important; }

.health-card {
    border-radius:16px; padding:20px 24px; margin-bottom:16px;
    display:flex; align-items:center; gap:20px; border: 2px solid;
}
.health-healthy  { background:#f0faf4; border-color:#22c55e; }
.health-moderate { background:#fffbeb; border-color:#f59e0b; }
.health-unhealthy{ background:#fff1f2; border-color:#ef4444; }
.health-grade { font-size:48px; font-weight:900; line-height:1; min-width:56px; text-align:center; }
.health-healthy   .health-grade { color:#16a34a; }
.health-moderate  .health-grade { color:#d97706; }
.health-unhealthy .health-grade { color:#dc2626; }
.health-body { flex:1; }
.health-verdict { font-size:18px; font-weight:700; margin-bottom:2px; }
.health-score-line { font-size:13px; color:#64748b; margin-bottom:8px; }
.health-bar-wrap { background:#e2e8f0; border-radius:99px; height:8px; overflow:hidden; margin-bottom:10px; }
.health-bar { height:8px; border-radius:99px; transition: width .4s; }
.health-healthy   .health-bar { background: linear-gradient(90deg,#22c55e,#16a34a); }
.health-moderate  .health-bar { background: linear-gradient(90deg,#fbbf24,#d97706); }
.health-unhealthy .health-bar { background: linear-gradient(90deg,#f87171,#dc2626); }
.health-tags { display:flex; flex-wrap:wrap; gap:6px; }
.htag { font-size:11px; padding:2px 9px; border-radius:10px; font-weight:600; }
.htag-neg { background:#fee2e2; color:#991b1b; }
.htag-pos { background:#dcfce7; color:#166534; }
.section-scores { display:flex; gap:10px; margin-top:10px; flex-wrap:wrap; }
.sscore { font-size:11px; background:#f1f5f9; border-radius:8px; padding:4px 10px; color:#475569; font-weight:600; }

.diet-card {
    border-radius: 14px; padding: 14px 18px; margin-bottom: 10px;
    border-left: 5px solid; position: relative; transition: box-shadow .15s;
}
.diet-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.08); }
.diet-yes    { background:#f0faf4; border-color:#22c55e; }
.diet-no     { background:#fff1f2; border-color:#ef4444; }
.diet-caution{ background:#fffbeb; border-color:#f59e0b; }
.diet-uncertain{ background:#f0f4ff; border-color:#6366f1; }
.diet-icon   { font-size:22px; margin-right:8px; vertical-align:middle; }
.diet-title  { font-size:15px; font-weight:700; vertical-align:middle; }
.diet-label  { font-size:12px; font-weight:600; margin-left:6px;
               padding:1px 8px; border-radius:10px; vertical-align:middle; }
.diet-yes   .diet-label  { background:#dcfce7; color:#166534; }
.diet-no    .diet-label  { background:#fee2e2; color:#991b1b; }
.diet-caution .diet-label{ background:#fef3c7; color:#92400e; }
.diet-uncertain .diet-label{ background:#e0e7ff; color:#3730a3; }
.diet-reason { font-size:12px; color:#64748b; margin-top:5px; padding-left:32px; }
.diet-reason .neg { color:#dc2626; margin-right:6px; }
.diet-reason .pos { color:#16a34a; margin-right:6px; }

/* Smart pairing cards */
.pair-card {
    border-radius:14px; padding:16px 20px; margin-bottom:12px;
    background:#f8faff; border:1.5px solid #c7d5f5;
    transition: box-shadow .15s;
}
.pair-card:hover { box-shadow:0 4px 16px rgba(0,0,0,0.09); }
.pair-rank { font-size:22px; font-weight:900; color:#1a56db; margin-right:10px; }
.pair-name { font-size:15px; font-weight:700; color:#1e293b; }
.pair-cat  { font-size:11px; color:#64748b; background:#e8f0fe;
             border-radius:8px; padding:2px 8px; margin-left:8px; }
.pair-score { font-size:12px; color:#64748b; margin-top:4px; }
.pair-tier { font-size:11px; font-weight:700; background:#dbeafe;
             color:#1d4ed8; border-radius:6px; padding:1px 7px; margin-right:6px; }
.pair-bar-wrap { background:#e2e8f0; border-radius:99px; height:6px;
                 overflow:hidden; margin:6px 0; }
.pair-bar { height:6px; border-radius:99px;
            background:linear-gradient(90deg,#6366f1,#1d4ed8); }
.gap-table { font-size:12px; width:100%; border-collapse:collapse; margin-top:8px; }
.gap-table th { font-weight:600; color:#475569; padding:2px 8px; text-align:left; }
.gap-table td { padding:2px 8px; color:#1e293b; }
.gap-filled-pos { color:#16a34a; font-weight:600; }
.gap-filled-neg { color:#dc2626; }
</style>
""", unsafe_allow_html=True)


# ── Session-state initialisation ─────────────────────────────────────────────
DEFAULTS = dict(
    result       = None,
    dri          = None,
    sb_name      = "",
    sb_allergen  = "",
    sb_sex       = "Female",
    sb_age       = 25,
    sb_height    = 165.0,
    sb_weight    = 60.0,
    sb_activity  = "Active",
    sb_pregnancy = "None",
)
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Load food DB + detection data (cached) ────────────────────────────────────
@st.cache_resource
def _load_pairing_data():
    """Load food DB and detection lookups once."""
    data_dir = Path(__file__).parent / "data"

    food_db = pd.read_csv(data_dir / "food_products_cleaned.csv")

    allergen_master    = pd.read_csv(data_dir / "allergen_master.csv")
    allergen_tokens    = pd.read_csv(data_dir / "allergen_tokens.csv")
    ingredient_sources = pd.read_csv(data_dir / "ingredient_sources.csv")
    additives_df       = pd.read_csv(data_dir / "japanese_food_additives.csv", on_bad_lines="skip")

    id_to_allergen = dict(zip(
        allergen_master["allergen_id"].astype(int),
        allergen_master["canonical_name"].str.lower().str.strip(),
    ))
    token_to_allergen_id: dict[str, int] = {}
    for _, row in allergen_tokens.iterrows():
        if pd.notna(row["allergen_id"]) and pd.notna(row["token"]):
            token_to_allergen_id[str(row["token"]).lower().strip()] = int(row["allergen_id"])
    for _, row in ingredient_sources.iterrows():
        if pd.notna(row["allergen_id"]) and pd.notna(row["ingredient"]):
            key = str(row["ingredient"]).lower().strip()
            if key not in token_to_allergen_id:
                token_to_allergen_id[key] = int(row["allergen_id"])

    additives_df["name_normalized"] = additives_df["name"].str.lower().str.strip()

    with open(data_dir / "diet_blocklists.json", encoding="utf-8") as f:
        diet_lists = json.load(f)

    return food_db, token_to_allergen_id, id_to_allergen, additives_df, diet_lists


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🍱 Food Label Scanner")
    st.divider()

    st.markdown("### 👤 Your Profile")
    st.session_state.sb_name = st.text_input(
        "Name", value=st.session_state.sb_name, placeholder="e.g. Anshika"
    )
    st.session_state.sb_allergen = st.text_input(
        "Your allergen (optional)",
        value=st.session_state.sb_allergen,
        placeholder="e.g. peanut, milk, wheat",
        help="Allergen you are sensitive to — leave blank if none.",
    ).strip().lower()

    st.divider()

    st.markdown("### 🧮 DRI Calculator")
    st.caption("USDA Dietary Reference Intakes")

    c1, c2 = st.columns(2)
    with c1:
        st.session_state.sb_sex = st.selectbox(
            "Sex", ["Female", "Male"],
            index=["Female","Male"].index(st.session_state.sb_sex),
        )
    with c2:
        st.session_state.sb_age = st.number_input(
            "Age (yr)", min_value=19, max_value=100,
            value=st.session_state.sb_age, step=1,
        )

    c3, c4 = st.columns(2)
    with c3:
        st.session_state.sb_height = st.number_input(
            "Height (cm)", min_value=100.0, max_value=250.0,
            value=st.session_state.sb_height, step=0.5,
        )
    with c4:
        st.session_state.sb_weight = st.number_input(
            "Weight (kg)", min_value=20.0, max_value=300.0,
            value=st.session_state.sb_weight, step=0.5,
        )

    activity_opts = ["Sedentary", "Low Active", "Active", "Very Active"]
    st.session_state.sb_activity = st.selectbox(
        "Activity Level", activity_opts,
        index=activity_opts.index(st.session_state.sb_activity),
    )

    preg_opts = ["None","1st trimester","2nd trimester","3rd trimester",
                 "Breastfeeding 0-6 months","Breastfeeding 7-12 months"]
    st.session_state.sb_pregnancy = st.selectbox(
        "Pregnancy / Breastfeeding", preg_opts,
        index=preg_opts.index(st.session_state.sb_pregnancy),
    )

    if st.button("⚡ Calculate DRI", type="primary", use_container_width=True):
        pg = st.session_state.sb_pregnancy
        pregnancy = pg if "trimester"    in pg else "None"
        lactation = pg.replace("Breastfeeding ","") if "Breastfeeding" in pg else "None"
        st.session_state.dri = calculate_dri(
            sex       = st.session_state.sb_sex.lower(),
            age       = int(st.session_state.sb_age),
            weight_kg = float(st.session_state.sb_weight),
            height_cm = float(st.session_state.sb_height),
            activity  = st.session_state.sb_activity,
            pregnancy = pregnancy,
            lactation = lactation,
        )
        st.success("DRI calculated ✓")

    st.divider()
    st.caption("All scans are saved with your name, allergen, DRI profile, and detected results.")

    if st.session_state.dri:
        d = st.session_state.dri["inputs"]
        st.markdown(
            f"**Saved profile:**  \n"
            f"{d['sex'].title()} · {d['age']} yrs · {d['height_cm']} cm · "
            f"{d['weight_kg']} kg · {d['activity']}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def bmi_color(cls):
    return {"Underweight":"#3B9AE1","Normal weight":"#2EC4B6",
            "Overweight":"#F4A261","Obese":"#E63946"}.get(cls,"#888")

def confidence_badge(score):
    if score >= 80: return f"🟢 {score}%"
    if score >= 50: return f"🟡 {score}%"
    return f"🔴 {score}%"

def compute_confidence(jp, en):
    s = {}
    pname = (jp or {}).get("product_name")
    s["Product Name"] = 100 if pname and str(pname).strip() else 0
    ing   = (jp or {}).get("ingredients") or {}
    items = ing.get("items") or [] if isinstance(ing, dict) else []
    s["Ingredients"] = 100 if len(items)>=5 else 80 if len(items)>=2 else 70 if len(items)==1 else 0
    alg = (jp or {}).get("allergens")
    if   alg is None:        s["Allergens"] = 0
    elif isinstance(alg, dict):
        st2, ai = alg.get("style"), alg.get("items") or []
        s["Allergens"] = 100 if st2 and ai else 80 if st2 == "none" else 60 if st2 else 30
    else: s["Allergens"] = 30
    nut    = (jp or {}).get("nutrition") or {}
    filled = sum(1 for k in ["calories","protein","fat","carbohydrate","salt"] if nut.get(k))
    s["Nutrition"] = round(filled/5*100)
    return {"fields": s, "overall": round(sum(s.values())/len(s))}

_NUT_LIMITS = {
    "Calories":      (9999, None),
    "Protein":       (200,  None),
    "Fat":           (200,  None),
    "Saturated Fat": (100,  None),
    "Carbohydrate":  (500,  None),
    "Sugar":         (300,  None),
    "Fibre":         (100,  None),
    "Salt":          (10,   100),
}

def _strip_to_float(val):
    import re as _re
    if val is None:
        return None, False
    s = str(val).strip()
    if _re.fullmatch(r"0+\.?0*\s*%", s):
        return 0.0, False
    if "%" in s:
        return None, False
    is_mg = bool(_re.search(r"mg", s, _re.IGNORECASE))
    try:
        return float(s), is_mg
    except (ValueError, TypeError):
        m = _re.search(r"-?\d+\.?\d*", s)
        if m:
            try:
                return float(m.group()), is_mg
            except ValueError:
                pass
    return None, is_mg

def _clamp_nutrition(label, value, is_mg):
    unit = "kcal" if label == "Calories" else "g"
    if label == "Calories":
        return value, unit
    if is_mg:
        value = value / 1000.0
    elif label in _NUT_LIMITS:
        _, mg_thresh = _NUT_LIMITS[label]
        if mg_thresh and value > mg_thresh:
            value = value / 1000.0
    return round(value, 3), unit

def parse_nutrition(english):
    nut = (english or {}).get("nutrition") or {}
    if not nut: return None
    import re as _re2
    def _qn(v):
        if v is None: return None
        m = _re2.search(r'\d+\.?\d*', str(v))
        return float(m.group()) if m else None
    sugar_v = _qn(nut.get("sugar"))
    carb_v  = _qn(nut.get("carbohydrate"))
    if sugar_v is not None and carb_v is not None and abs(sugar_v - carb_v) < 0.1:
        nut["sugar"] = None
    def _get(nut, *keys):
        for k in keys:
            v = nut.get(k)
            if v is not None:
                return v
        return None
    field_map = [
        (["calories_kcal",    "calories"],               "Calories"),
        (["protein_g",        "protein"],                "Protein"),
        (["fat_g",            "fat"],                    "Fat"),
        (["saturated_fat_g",  "saturated_fat"],          "Saturated Fat"),
        (["carbohydrate_g",   "carbohydrate","carbs"],   "Carbohydrate"),
        (["sugar_g",          "sugar","sugars"],         "Sugar"),
        (["fibre_g",          "fibre","fiber","dietary_fibre","dietary_fiber"], "Fibre"),
        (["salt_g",           "salt","sodium_g","sodium"],"Salt"),
    ]
    rows = []
    for keys, label in field_map:
        val = _get(nut, *keys)
        if val is None: continue
        numeric, is_mg = _strip_to_float(val)
        if numeric is not None:
            corrected, unit = _clamp_nutrition(label, numeric, is_mg)
            rows.append({"Nutrient": label, "Amount": corrected, "Unit": unit})
        else:
            unit = "kcal" if label == "Calories" else "g"
            rows.append({"Nutrient": label, "Amount": str(val), "Unit": unit})
    return {"rows": rows, "basis": nut.get("basis")} if rows else None

def nutrition_pie(rows):
    keys = {"Protein","Fat","Carbohydrate"}
    labels,values = [],[]
    for r in rows:
        if r["Nutrient"] not in keys: continue
        try: v = float(r["Amount"])
        except: v = 0.0
        if v>0: labels.append(r["Nutrient"]); values.append(v)
    if not labels: return None
    colors = ["#4C9BE8","#F4A261","#2EC4B6"][:len(labels)]
    fig,ax = plt.subplots(figsize=(3.2,3.2),facecolor="none")
    _,_,autotexts = ax.pie(values,labels=None,autopct="%1.1f%%",colors=colors,
                           startangle=140,wedgeprops={"edgecolor":"white","linewidth":1.5})
    for at in autotexts: at.set_fontsize(10);at.set_color("white");at.set_fontweight("bold")
    patches = [mpatches.Patch(color=c,label=f"{l} ({v:.1f}g)") for c,l,v in zip(colors,labels,values)]
    ax.legend(handles=patches,loc="lower center",bbox_to_anchor=(0.5,-0.22),
              ncol=len(labels),fontsize=8,frameon=False)
    ax.set_title("Macronutrients",fontsize=10,pad=6)
    fig.tight_layout(); return fig

def dri_bar_chart(nut_rows, dri_raw):
    pairs = [("Calories","calories","kcal"),("Protein","protein_g","g"),
             ("Fat","fat_max_g","g"),("Carbohydrate","carb_max_g","g"),
             ("Sugar","sugar_limit_g","g"),("Salt","sodium_g","g")]
    lv_map = {r["Nutrient"]: r["Amount"] for r in nut_rows if isinstance(r.get("Amount"),float)}
    names,pcts = [],[]
    for nutrient,dri_key,_ in pairs:
        lv = lv_map.get(nutrient); dv = dri_raw.get(dri_key)
        if lv is not None and dv:
            try: names.append(nutrient); pcts.append(round(float(lv)/float(dv)*100,1))
            except: pass
    if not names: return None
    colors = ["#2EC4B6" if p<=25 else "#4C9BE8" if p<=50 else "#F4A261" if p<=80 else "#E63946"
              for p in pcts]
    fig,ax = plt.subplots(figsize=(4.5,3.5),facecolor="none")
    bars = ax.barh(names,pcts,color=colors,edgecolor="none",height=0.42)
    ax.axvline(100,color="#E63946",linestyle="--",linewidth=1.1,alpha=0.6,label="100% DRI")
    for bar,pct in zip(bars,pcts):
        ax.text(bar.get_width()+0.8,bar.get_y()+bar.get_height()/2,
                f"{pct}%",va="center",fontsize=9,color="#333")
    ax.set_xlabel("% of Daily DRI",fontsize=9)
    ax.set_xlim(0,max(pcts+[100])*1.28)
    ax.tick_params(labelsize=9)
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(fontsize=8,frameon=False)
    fig.tight_layout(); return fig


def _nutrition_output_from_result(result: dict) -> dict:
    """Convert scan result to the flat nutrition dict smart_pairing expects."""
    english = (result.get("ocr") or {}).get("english") or {}
    nut     = (english.get("nutrition")) or {}
    import re as _re

    def _n(v):
        if v is None: return None
        m = _re.search(r"\d+\.?\d*", str(v))
        return float(m.group()) if m else None

    return {
        "calories":        _n(nut.get("calories")),
        "protein_g":       _n(nut.get("protein")),
        "fat_g":           _n(nut.get("fat")),
        "carbohydrate_g":  _n(nut.get("carbohydrate")),
        "sugar_g":         _n(nut.get("sugar")),
        "fibre_g":         _n(nut.get("fibre")),
        "salt_g":          _n(nut.get("salt")),
        "saturated_fat_g": _n(nut.get("saturated_fat")),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_scan, tab_pair, tab_dri = st.tabs(["📸 Scan Label", "🍱 Smart Pairing", "🧮 DRI Profile"])


# ════════════════════════════════════════════════════
# TAB 1 — Scanner (unchanged layout, both columns)
# ════════════════════════════════════════════════════
with tab_scan:
    left_col, right_col = st.columns([11, 9], gap="large")

    with left_col:
        st.markdown("## 📸 Scan a Food Label")

        uploaded = st.file_uploader(
            "Drop a food packet photo here",
            type=["jpg","jpeg","png","webp"],
            label_visibility="collapsed",
        )
        if uploaded:
            st.image(uploaded, use_container_width=True)
            _sig = uploaded.name + str(uploaded.size)
            if st.session_state.get("_last_uploaded","") != _sig:
                st.session_state.result = None

        user_name     = st.session_state.sb_name.strip()
        user_allergen = st.session_state.sb_allergen

        scan_ready   = bool(uploaded and user_name)
        scan_clicked = st.button("🔍 Scan Label", type="primary", disabled=not scan_ready)

        if not uploaded:
            st.info("📂 Upload a Japanese food label image above.")
        elif not user_name:
            st.warning("✏️ Enter your name in the sidebar first.")

        if scan_clicked and scan_ready:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
                tmp.write(uploaded.read()); tmp_path = tmp.name

            with st.spinner("🔄 Running OCR + translation…"):
                from services.ocr import extract_label
                ocr_result = extract_label(tmp_path)

            if ocr_result.get("error") and not ocr_result.get("ingredients"):
                st.error(f"OCR failed: {ocr_result['error']}")
            else:
                ingredients_text = ocr_result.get("ingredients","")
                english  = ocr_result.get("english") or {}
                japanese = ocr_result.get("japanese") or {}

                with st.spinner("🔬 Detecting allergens & additives…"):
                    detection = analyze_ingredients(ingredients_text)

                en_nutrition  = (ocr_result.get("english") or {}).get("nutrition")
                diet_result   = classify_diet(
                    ingredients_flat = ingredients_text,
                    nutrition        = en_nutrition,
                    allergens        = detection["allergens"],
                )
                health_result = compute_health_score(
                    ingredients_flat = ingredients_text,
                    nutrition        = en_nutrition,
                    additives        = detection["additives"],
                    dri_raw          = (st.session_state.dri or {}).get("_raw"),
                )

                saved_img = save_image(tmp_path)
                user_id   = save_record(
                    user_id            = "",
                    user_name          = user_name,
                    user_allergen      = user_allergen,
                    image_path         = saved_img,
                    ingredients        = ingredients_text,
                    detected_allergens = detection["allergens"],
                    detected_additives = detection["additives"],
                    dri                = st.session_state.dri,
                )

                st.session_state.result = {
                    "user_id":       user_id,
                    "ocr":           ocr_result,
                    "detection":     detection,
                    "diet":          diet_result,
                    "health":        health_result,
                    "ingredients":   ingredients_text,
                    "user_allergen": user_allergen,
                    "confidence":    compute_confidence(japanese, english),
                }
                st.session_state._last_uploaded = uploaded.name + str(uploaded.size)

        # ── Show results ──────────────────────────────────────────────────────
        if st.session_state.result:
            r         = st.session_state.result
            detection = r["detection"]
            allergens = detection["allergens"]
            additives = detection["additives"]
            user_alg  = r["user_allergen"]
            ocr       = r["ocr"]
            english   = ocr.get("english") or {}
            japanese  = ocr.get("japanese") or {}
            conf      = r.get("confidence") or compute_confidence(japanese, english)

            st.divider()

            h1, h2, h3 = st.columns([3, 2, 2])
            with h1:
                pname = (english or {}).get("product_name") or "—"
                st.markdown(f"**{pname}**")
                st.caption(f"Scan `{r['user_id']}` · {ocr.get('latency_s','—')}s")
            with h2:
                st.metric("OCR Confidence", confidence_badge(conf["overall"]))
            with h3:
                total_flags = len(allergens) + len(additives)
                st.metric("Flags", f"{total_flags} item{'s' if total_flags!=1 else ''}")

            with st.expander("📊 OCR Confidence Breakdown", expanded=False):
                ccols = st.columns(len(conf["fields"]))
                for col,(field,score) in zip(ccols, conf["fields"].items()):
                    col.metric(field, confidence_badge(score))
                st.caption("🟢 ≥80% · 🟡 50-79% · 🔴 <50%")

            with st.expander("🧾 Ingredients (English)", expanded=True):
                raw_items = english.get("ingredients") or []
                if isinstance(raw_items, list) and raw_items:
                    chips = "".join(f'<span class="chip">{item}</span>' for item in raw_items)
                    st.markdown(chips, unsafe_allow_html=True)
                elif r["ingredients"]:
                    st.write(r["ingredients"])
                else:
                    st.info("No ingredients extracted.")

            st.markdown('<p class="sec-header">⚠️ Allergen & Additive Alerts</p>', unsafe_allow_html=True)

            user_alg_found = any(
                (user_alg in det or det in user_alg)
                for det in allergens
            ) if user_alg else False

            if user_alg and user_alg_found:
                st.error(f"🚨 **Your allergen detected: {user_alg.title()}** — this product contains it.", icon="🚨")

            mandatory   = [a for a in allergens if a in JP_MANDATORY]
            other       = [a for a in allergens if a not in JP_MANDATORY]
            recommended = [a for a in other if not (user_alg and (user_alg in a or a in user_alg))]

            acol1, acol2 = st.columns(2)
            with acol1:
                st.markdown("**🏷️ Allergens**")
                if mandatory:
                    st.markdown("🔴 Mandatory (JP law)")
                    st.markdown(
                        "".join(f'<span class="pill pill-red">{a.title()}</span>' for a in sorted(mandatory)),
                        unsafe_allow_html=True,
                    )
                if recommended:
                    st.markdown("🟡 Recommended")
                    st.markdown(
                        "".join(f'<span class="pill pill-yellow">{a.title()}</span>' for a in sorted(recommended)),
                        unsafe_allow_html=True,
                    )
                if not mandatory and not recommended and not user_alg_found:
                    st.markdown('<span class="pill pill-green">✅ None detected</span>', unsafe_allow_html=True)

            with acol2:
                st.markdown("**🧪 Additives**")
                if additives:
                    st.markdown(
                        "".join(
                            f'<span class="pill pill-orange">{aname}'
                            f'<span style="font-size:10px;opacity:.7"> #{aid}</span></span>'
                            for aid,aname in additives
                        ),
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown('<span class="pill pill-green">✅ None detected</span>', unsafe_allow_html=True)

            health = r.get("health") or {}
            if health:
                st.markdown('<p class="sec-header">🏥 Health Score</p>', unsafe_allow_html=True)
                score   = health.get('score', 0)
                verdict = health.get('verdict', 'Moderate')
                grade   = health.get('grade', 'C')
                secs    = health.get('sections', {})
                t_flags = health.get('top_flags', [])
                t_boost = health.get('top_boosts', [])
                vcls    = {'Healthy':'health-healthy','Moderate':'health-moderate','Unhealthy':'health-unhealthy'}.get(verdict,'health-moderate')

                tags_html = ''
                for f in t_flags[:3]: tags_html += f'<span class="htag htag-neg">✗ {f}</span>'
                for b in t_boost[:2]: tags_html += f'<span class="htag htag-pos">✓ {b}</span>'
                sec_html = ''
                sec_labels = {'ingredients':'Ingredients','macros':'Macros','sodium':'Sodium','additives':'Additives'}
                for sk, sv in secs.items():
                    sec_html += f'<span class="sscore">{sec_labels.get(sk,sk).title()} {sv["score"]}/{sv["max"]}</span>'

                st.markdown(f'''
                <div class="health-card {vcls}">
                    <div class="health-grade">{grade}</div>
                    <div class="health-body">
                        <div class="health-verdict">{verdict}</div>
                        <div class="health-score-line">Score: {score}/100</div>
                        <div class="health-bar-wrap"><div class="health-bar" style="width:{score}%"></div></div>
                        <div class="health-tags">{tags_html}</div>
                        <div class="section-scores">{sec_html}</div>
                    </div>
                </div>
                ''', unsafe_allow_html=True)

            diet = r.get("diet") or {}
            if diet:
                st.markdown('<p class="sec-header">🥗 Diet Classification</p>', unsafe_allow_html=True)
                DIET_META = {
                    "vegan":      ("🌱", "Vegan"),
                    "vegetarian": ("🥚", "Vegetarian"),
                    "keto":       ("🥑", "Keto"),
                    "diabetic":   ("🩺", "Diabetic-Friendly"),
                }
                STATUS_CLASS = {
                    "yes": "diet-yes", "no": "diet-no",
                    "caution": "diet-caution", "uncertain": "diet-uncertain",
                }
                dcols = st.columns(2)
                for idx, (key, (icon, title)) in enumerate(DIET_META.items()):
                    d = diet.get(key, {})
                    status    = d.get("status", "uncertain")
                    label     = d.get("label", "Unknown")
                    reasons   = d.get("reasons", [])
                    positives = d.get("positives", [])
                    css_cls   = STATUS_CLASS.get(status, "diet-uncertain")
                    reason_html = ""
                    for r_txt in reasons[:2]:
                        reason_html += f'<span class="neg">✗</span>{r_txt}<br>'
                    for p_txt in positives[:1]:
                        reason_html += f'<span class="pos">✓</span>{p_txt}<br>'
                    html = (
                        f'<div class="diet-card {css_cls}">'
                        f'<span class="diet-icon">{icon}</span>'
                        f'<span class="diet-title">{title}</span>'
                        f'<span class="diet-label">{label}</span>'
                        f'<div class="diet-reason">{reason_html}</div>'
                        f'</div>'
                    )
                    with dcols[idx % 2]:
                        st.markdown(html, unsafe_allow_html=True)

            nut_data = parse_nutrition(english)
            if nut_data and nut_data["rows"]:
                st.markdown('<p class="sec-header">🥗 Nutrition Information</p>', unsafe_allow_html=True)
                if nut_data["basis"]:
                    st.caption(f"Per serving: {nut_data['basis']}")
                nc1, nc2 = st.columns([1,1])
                with nc1:
                    df = pd.DataFrame(nut_data["rows"])
                    def _fmt_value(row):
                        amt, unit = row["Amount"], row["Unit"]
                        if isinstance(amt, float):
                            n = int(amt) if amt == int(amt) else round(amt, 2)
                            return f"{n} {unit}"
                        return str(amt)
                    df["Value"] = df.apply(_fmt_value, axis=1)
                    st.dataframe(df[["Nutrient","Value"]].set_index("Nutrient"), use_container_width=True)
                with nc2:
                    fig = nutrition_pie(nut_data["rows"])
                    if fig: st.pyplot(fig, use_container_width=True)
                    else:   st.info("No macronutrient data for chart.")

                if st.session_state.dri:
                    st.markdown('<p class="sec-header">📊 This Product vs Your Daily DRI</p>', unsafe_allow_html=True)
                    st.caption("What % of your estimated daily needs does this single product provide?")
                    comp_fig = dri_bar_chart(nut_data["rows"], st.session_state.dri["_raw"])
                    if comp_fig:
                        st.pyplot(comp_fig, use_container_width=True)
                        dri_raw   = st.session_state.dri["_raw"]
                        lv_map    = {r["Nutrient"]: r["Amount"] for r in nut_data["rows"]
                                     if isinstance(r.get("Amount"), float)}
                        comp_rows = []
                        sodium_g     = dri_raw.get("sodium_g")
                        sodium_label = f"{sodium_g} g" if sodium_g else "1.5 g"
                        sugar_lim    = dri_raw.get("sugar_limit_g")
                        sugar_label  = f"{sugar_lim} g (WHO 10%)" if sugar_lim else "—"
                        for nutrient,dkey,dlabel in [
                            ("Calories",    "calories",       f"{dri_raw.get('calories')} kcal"),
                            ("Protein",     "protein_g",      f"{dri_raw.get('protein_g')} g"),
                            ("Fat",         "fat_max_g",      f"{dri_raw.get('fat_max_g')} g max"),
                            ("Carbohydrate","carb_max_g",     f"{dri_raw.get('carb_max_g')} g max"),
                            ("Sugar",       "sugar_limit_g",  sugar_label),
                            ("Salt",        "sodium_g",       sodium_label),
                        ]:
                            lv = lv_map.get(nutrient); dv = dri_raw.get(dkey)
                            if lv is not None and dv:
                                try:
                                    pct = round(float(lv)/float(dv)*100, 1)
                                    comp_rows.append({"Nutrient":nutrient,"In Product":f"{lv}",
                                                      "Daily DRI":dlabel,"% of Need":f"{pct}%"})
                                except: pass
                        if comp_rows:
                            st.dataframe(pd.DataFrame(comp_rows).set_index("Nutrient"),
                                         use_container_width=True)
                else:
                    st.caption("💡 Calculate your DRI in the sidebar to compare this product against your daily needs.")
            else:
                st.caption("ℹ️ No nutrition data extracted from this label.")

            with st.expander("🔬 Raw OCR output (Japanese)", expanded=False):
                st.json(japanese or {})
                s1_raw = ocr.get("s1_raw") or ""
                s2_raw = ocr.get("s2_raw") or ""
                st.caption("Stage 1 raw (OCR — Japanese):")
                st.code(s1_raw[:3000] if s1_raw else "(empty)", language="json")
                st.caption("Stage 2 raw (Translation — English):")
                st.code(s2_raw[:3000] if s2_raw else "(empty)", language="json")

    # ── Right col: DRI profile summary (scan tab) ─────────────────────────────
    with right_col:
        st.markdown("## 🧮 Your DRI Profile")
        if not st.session_state.dri:
            st.markdown("""
            <div style="background:#f0f4ff;border-radius:12px;padding:20px 24px;
                        border:1px solid #c7d5f5;color:#2d3a6e;font-size:14px">
              <b>Fill in your details in the sidebar</b> and click
              <b>⚡ Calculate DRI</b> to see your personalised
              Dietary Reference Intake values.
            </div>
            """, unsafe_allow_html=True)
        else:
            dri  = st.session_state.dri
            inp  = dri["inputs"]
            bclr = bmi_color(dri["bmi_class"])
            st.markdown(
                f'<div style="background:#eef6ff;border-radius:10px;padding:10px 16px;'
                f'font-size:13px;color:#1e3a5f;margin-bottom:12px">'
                f'<b>{inp["sex"].title()}</b> · {inp["age"]} yrs · '
                f'{inp["height_cm"]} cm · {inp["weight_kg"]} kg · '
                f'<b>{inp["activity"]}</b>'
                f'</div>',
                unsafe_allow_html=True,
            )
            m1, m2 = st.columns(2)
            with m1:
                st.metric("BMI", dri["bmi"])
                st.markdown(
                    f'<span style="background:{bclr};color:white;border-radius:10px;'
                    f'padding:2px 10px;font-size:12px;font-weight:600">{dri["bmi_class"]}</span>',
                    unsafe_allow_html=True,
                )
            with m2:
                st.metric("Daily Calories", f"{dri['eer']:,} kcal")
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("🥦 Macronutrients", expanded=True):
                rows = [{"Nutrient":k,"Recommended / Day":v["value"],"Basis":v["note"]}
                        for k,v in dri["macros"].items()]
                st.dataframe(pd.DataFrame(rows).set_index("Nutrient"), use_container_width=True)
            with st.expander("💊 Vitamins", expanded=False):
                rows = []
                for name,vals in dri["vitamins"].items():
                    ul = str(vals["ul"]) if vals["ul"] != "ND" else "—"
                    rows.append({"Vitamin":name,"RDA/AI":vals["rda"],"UL":ul})
                st.dataframe(pd.DataFrame(rows).set_index("Vitamin"), use_container_width=True)
            with st.expander("⚗️ Minerals", expanded=False):
                rows = []
                for name,vals in dri["minerals"].items():
                    ul = str(vals["ul"]) if vals["ul"] != "ND" else "—"
                    rows.append({"Mineral":name,"RDA/AI":vals["rda"],"UL":ul})
                st.dataframe(pd.DataFrame(rows).set_index("Mineral"), use_container_width=True)


# ════════════════════════════════════════════════════
# TAB 2 — Smart Pairing
# ════════════════════════════════════════════════════
with tab_pair:
    st.markdown("## 🍱 Smart Pairing")
    st.caption("Find food products from the database that best complement your scanned product's nutrition toward a balanced meal.")

    if not st.session_state.result:
        st.info("📸 Scan a food label first in the **Scan Label** tab, then come back here.")
    elif not st.session_state.dri:
        st.warning("🧮 Calculate your DRI profile in the sidebar first — pairing needs your daily calorie and nutrient targets.")
    else:
        r          = st.session_state.result
        dri_raw    = st.session_state.dri["_raw"]
        ocr        = r["ocr"]
        english    = ocr.get("english") or {}
        diet       = r.get("diet") or {}
        user_alg   = r.get("user_allergen", "")

        scanned_nutrition = _nutrition_output_from_result(r)
        product_name      = (english.get("product_name") or "Scanned product")

        # ── Controls ─────────────────────────────────────────────────────────
        # REPLACE WITH:
        pc1, pc2, pc3 = st.columns([3, 2, 2])
        with pc1:
            st.markdown(f"**Scanned product:** {product_name}")
        with pc2:
            top_n = st.selectbox("Top suggestions", [3, 5, 10], index=1)
        with pc3:
            meals = st.selectbox("Meals per day", [2, 3, 4], index=1,
                                 help="Divides daily DRI by this number to get per-meal targets.")

        # User allergen exclusion
        user_allergens_input = st.text_input(
            "Exclude allergens (comma-separated)",
            value=user_alg,
            placeholder="e.g. milk, wheat, egg",
            help="Products containing these will be excluded from suggestions.",
        )
        user_allergens = [a.strip().lower() for a in user_allergens_input.split(",") if a.strip()]

        # ── Brand + Category filters ──────────────────────────────────────────
        food_db, token_to_allergen_id, id_to_allergen, additives_df, diet_lists = _load_pairing_data()

        all_categories = sorted(food_db["Category"].dropna().unique().tolist())
        all_brands = sorted(set(
            str(n).split()[0]          # first word of Name-EN as brand proxy
            for n in food_db["Name-EN"].dropna()
            if str(n).strip()
        ))

        with st.expander("🔎 Filter by Category & Brand (optional)", expanded=False):
            fc1, fc2 = st.columns(2)
            with fc1:
                selected_categories = st.multiselect(
                    "Categories",
                    options=all_categories,
                    default=[],
                    placeholder="All categories",
                    help="Leave empty to include all categories.",
                )
            with fc2:
                selected_brands = st.multiselect(
                    "Brands",
                    options=all_brands,
                    default=[],
                    placeholder="All brands",
                    help="Filters by first word of product name. Leave empty for all brands.",
                )
            st.caption(
                "💡 Use this to focus suggestions — e.g. pick 'SNACKS' + 'CEREAL' "
                "for snack-time pairings, or select specific brands you trust."
            )

        run_pairing = st.button("🔍 Find Pairings", type="primary")

        # REPLACE WITH:
        if run_pairing:
            with st.spinner("Scoring database products…"):
                # Apply category + brand filters to DB before scoring
                filtered_db = food_db.copy()
                if selected_categories:
                    filtered_db = filtered_db[
                        filtered_db["Category"].isin(selected_categories)
                    ]
                if selected_brands:
                    filtered_db = filtered_db[
                        filtered_db["Name-EN"].apply(
                            lambda n: any(
                                str(n).startswith(b) for b in selected_brands
                            )
                        )
                    ]

                if filtered_db.empty:
                    st.warning("No products match the selected filters. Try broadening your selection.")
                    st.stop()

                # Score top 30 first, then apply diversity filter for display
                pairing = smart_pair(
                    scanned_nutrition    = scanned_nutrition,
                    scanned_diet         = diet,
                    dri_raw              = dri_raw,
                    food_db              = filtered_db,
                    token_to_allergen_id = token_to_allergen_id,
                    id_to_allergen       = id_to_allergen,
                    additives_df         = additives_df,
                    diet_lists           = diet_lists,
                    user_allergens       = user_allergens,
                    top_n                = 30,           # fetch more, diversity-filter below
                    meals                = meals,
                    product_name         = product_name,
                )

                # Diversity filter: max 2 per brand (first word), keep top_n overall
                seen_brands: dict[str, int] = {}
                diverse = []
                for c in pairing["top_pairings"]:
                    brand = str(c.get("name_en","")).split()[0] if c.get("name_en") else ""
                    count = seen_brands.get(brand, 0)
                    if count < 2:
                        diverse.append(c)
                        seen_brands[brand] = count + 1
                    if len(diverse) >= top_n:
                        break
                pairing["top_pairings"] = diverse
                pairing["_filter_note"] = (
                    f"Categories: {', '.join(selected_categories) or 'All'}  |  "
                    f"Brands: {', '.join(selected_brands) or 'All'}  |  "
                    f"Showing top {len(diverse)} (diversity-filtered from {len(pairing['top_pairings'])+len(diverse)} scored)"
                )
                st.session_state["pairing"] = pairing

        # REPLACE WITH:
        if "pairing" in st.session_state:
            pairing = st.session_state["pairing"]

            # Filter summary
            if pairing.get("_filter_note"):
                st.caption(f"🔎 {pairing['_filter_note']}")

            # ── Nutrient gap summary ──────────────────────────────────────────
            st.markdown("---")
            st.markdown("### 📊 Nutrient Gap After Scanned Product")
            st.caption(f"Per-meal target = daily DRI ÷ {meals} meals")

            gap      = pairing["gap_after_scan"]
            target   = pairing["meal_dri_target"]
            gap_cols = st.columns(len(gap))
            icons_map = {
                "calories":"🔥","protein_g":"💪","fat_g":"🧈",
                "carbohydrate_g":"🌾","salt_g":"🧂",
            }
            for col, (k, v) in zip(gap_cols, gap.items()):
                tgt = target.get(k, 1)
                pct = round((1 - v / tgt) * 100) if tgt else 0
                col.metric(
                    f"{icons_map.get(k,'')} {k.replace('_g','').replace('_',' ').title()}",
                    f"{v:.1f}g" if "calorie" not in k else f"{v:.0f}kcal",
                    f"{pct}% filled",
                    delta_color="normal",
                )

            st.markdown(f"**{pairing['excluded_count']}** products excluded "
                        f"(allergen conflict or excessive salt).")

            # ── Pairing results ───────────────────────────────────────────────
            st.markdown("### 🏆 Top Pairing Suggestions")

            if not pairing["top_pairings"]:
                st.warning("No eligible pairing products found. Try removing allergen exclusions or adjusting meal count.")
            else:
                for rank, p in enumerate(pairing["top_pairings"], 1):
                    sb        = p["score_breakdown"]
                    composite = sb["composite"]
                    tier      = p.get("pareto_tier", "?")
                    pct_bar   = max(0, min(100, int(composite * 100)))

                    allergens_all = sorted(set(
                        p["allergens"]["in_product"] + p["allergens"]["cross_contact"]
                    ))
                    additives_hi = [a["name"] for a in p["additives"]
                                    if str(a.get("risk_level","")).lower() in ("high","medium")]

                    # Gap fill table rows
                    gap_rows = ""
                    for n in ["calories","protein_g","fat_g","carbohydrate_g","salt_g"]:
                        db_v   = p["nutrition"].get(n)
                        before = p["gap_before"].get(n, 0.0)
                        after  = p["gap_after"].get(n, 0.0)
                        filled = round(before - after, 2)
                        db_str = f"{db_v:.1f}" if db_v is not None else "—"
                        label  = n.replace("_g","").replace("_"," ").title()
                        cls    = "gap-filled-pos" if filled > 0 else "gap-filled-neg"
                        gap_rows += (
                            f"<tr><td>{label}</td><td>{db_str}</td>"
                            f'<td class="{cls}">{filled:+.1f}</td></tr>'
                        )

                    # Allergen / additive summary line
                    alg_str = ", ".join(a.title() for a in allergens_all) if allergens_all else "None"
                    add_str = ", ".join(additives_hi) if additives_hi else "None flagged"

                    html = f"""
                    <div class="pair-card">
                      <div>
                        <span class="pair-rank">#{rank}</span>
                        <span class="pair-name">{p['name_en']}</span>
                        <span class="pair-cat">{p['category']}</span>
                      </div>
                      <div class="pair-score">
                        <span class="pair-tier">Pareto T{tier}</span>
                        Composite score: <b>{composite:.3f}</b> &nbsp;·&nbsp;
                        L2={sb['l2_nutrition']:.3f} &nbsp;
                        Diet+={sb['diet_bonus']:.2f} &nbsp;
                        Add−={sb['additive_penalty']:.3f} &nbsp;
                        Salt−={sb['salt_penalty']:.3f}
                      </div>
                      <div class="pair-bar-wrap">
                        <div class="pair-bar" style="width:{pct_bar}%"></div>
                      </div>
                      <table class="gap-table">
                        <tr>
                          <th>Nutrient</th><th>DB value (g/kcal)</th><th>Gap filled</th>
                        </tr>
                        {gap_rows}
                      </table>
                      <div style="margin-top:8px;font-size:12px;color:#475569;">
                        🏷️ <b>Allergens:</b> {alg_str} &nbsp;&nbsp;
                        🧪 <b>Flagged additives:</b> {add_str}
                      </div>
                    </div>
                    """
                    st.markdown(html, unsafe_allow_html=True)

                # ── Download pairing JSON ─────────────────────────────────────
                st.download_button(
                    "⬇️ Download pairing result (JSON)",
                    data=json.dumps(pairing, ensure_ascii=False, indent=2),
                    file_name="smart_pairing_result.json",
                    mime="application/json",
                )


# ════════════════════════════════════════════════════
# TAB 3 — Full DRI Profile
# ════════════════════════════════════════════════════
with tab_dri:
    st.markdown("## 🧮 Your Full DRI Profile")
    if not st.session_state.dri:
        st.markdown("""
        <div style="background:#f0f4ff;border-radius:12px;padding:20px 24px;
                    border:1px solid #c7d5f5;color:#2d3a6e;font-size:14px">
          <b>Fill in your details in the sidebar</b> and click
          <b>⚡ Calculate DRI</b> to see your personalised
          Dietary Reference Intake values — calories, macros,
          vitamins, and minerals.
        </div>
        """, unsafe_allow_html=True)
    else:
        dri  = st.session_state.dri
        inp  = dri["inputs"]
        bclr = bmi_color(dri["bmi_class"])

        st.markdown(
            f'<div style="background:#eef6ff;border-radius:10px;padding:10px 16px;'
            f'font-size:13px;color:#1e3a5f;margin-bottom:12px">'
            f'<b>{inp["sex"].title()}</b> · {inp["age"]} yrs · '
            f'{inp["height_cm"]} cm · {inp["weight_kg"]} kg · '
            f'<b>{inp["activity"]}</b>'
            f'</div>',
            unsafe_allow_html=True,
        )

        m1, m2 = st.columns(2)
        with m1:
            st.metric("BMI", dri["bmi"])
            st.markdown(
                f'<span style="background:{bclr};color:white;border-radius:10px;'
                f'padding:2px 10px;font-size:12px;font-weight:600">{dri["bmi_class"]}</span>',
                unsafe_allow_html=True,
            )
        with m2:
            st.metric("Daily Calories", f"{dri['eer']:,} kcal")

        st.markdown("<br>", unsafe_allow_html=True)

        with st.expander("🥦 Macronutrients", expanded=True):
            rows = [{"Nutrient":k,"Recommended / Day":v["value"],"Basis":v["note"]}
                    for k,v in dri["macros"].items()]
            st.dataframe(pd.DataFrame(rows).set_index("Nutrient"), use_container_width=True)

        with st.expander("💊 Vitamins", expanded=False):
            rows = []
            for name,vals in dri["vitamins"].items():
                ul = str(vals["ul"]) if vals["ul"] != "ND" else "—"
                rows.append({"Vitamin":name,"RDA/AI":vals["rda"],"UL":ul})
            st.dataframe(pd.DataFrame(rows).set_index("Vitamin"), use_container_width=True)

        with st.expander("⚗️ Minerals", expanded=False):
            rows = []
            for name,vals in dri["minerals"].items():
                ul = str(vals["ul"]) if vals["ul"] != "ND" else "—"
                rows.append({"Mineral":name,"RDA/AI":vals["rda"],"UL":ul})
            st.dataframe(pd.DataFrame(rows).set_index("Mineral"), use_container_width=True)

        st.markdown(
            '<div style="background:#f0faf4;border-radius:8px;padding:10px 14px;'
            'font-size:12px;color:#1a5e36;margin-top:8px">'
            '✅ <b>This profile is saved with every scan</b> — sex, age, height, '
            'weight, activity, BMI, and estimated daily calories.'
            '</div>',
            unsafe_allow_html=True,
        )