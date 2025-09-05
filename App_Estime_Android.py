# -*- coding: utf-8 -*-
"""
Estimation de surfaces - Vinci Construction SN (version nettoyée)
-----------------------------------------------------------------
Application Streamlit pour :
- Sélection d’un tronçon PR→PR (à partir d’un fichier PR CSV/Excel)
- Édition de sous-segments sur carte (Folium/Leaflet)
- Calcul de distances et surfaces par profils (mix possible)
- Rabotage (multi-hauteurs) et Reprofilage (multi-matériaux)

Cette version conserve la logique métier d’origine et ajoute :
- Imports organisés, docstrings et commentaires explicatifs
- Petites améliorations de lisibilité et corrections mineures
  (parse des polylignes, légendes PR, ajout cercles, factorisation distance)

Auteur : Sofienn NASRI (code initial) / Nettoyage : Copilot
"""
from __future__ import annotations

# ─────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────
import os
import sys
import inspect
from math import hypot
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import streamlit as st
import folium
from folium import IFrame
from folium.plugins import Draw
from folium.features import DivIcon
from streamlit_folium import st_folium

from pyproj import Transformer
from PIL import Image  # conservé si besoin pour d’autres usages
import base64
from pathlib import Path
# ─────────────────────────────────────────────────────────────
# CONFIG APP & RESSOURCES
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Estimation Mobile", layout="wide", initial_sidebar_state="collapsed")
import base64
from pathlib import Path
import streamlit as st

def resource_path(relative_path: str) -> str:
    """Retourne le chemin absolu (compatible PyInstaller)."""
    import os, sys
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

def add_bg_image(image_path: str,
                 overlay_opacity: float = 0.03,  # Presque invisible
                 fixed: bool = True,
                 size: str = "auto",
                 position: str = "center center"):
    """Ajoute une image de fond à l'app Streamlit."""
    try:
        p = Path(resource_path(image_path))
        img_bytes = p.read_bytes()
    except Exception as e:
        st.warning(f"Impossible de lire l'image '{image_path}': {e}")
        return

    b64 = base64.b64encode(img_bytes).decode()
    fixed_flag = "fixed" if fixed else "scroll"
    overlay = f"linear-gradient(rgba(255,255,255,{overlay_opacity}), rgba(255,255,255,{overlay_opacity})), "

    css = f"""
    <style>
      [data-testid="stAppViewContainer"] {{
        background: {overlay} url("data:image/png;base64,{b64}") no-repeat {position} {fixed_flag};
        background-size: {size};
      }}
      [data-testid="stHeader"] {{
        background: rgba(255,255,255,0.0);
      }}
      [data-testid="stSidebar"] {{
        background: rgba(255,255,255,0.85);
        backdrop-filter: blur(2px);
      }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Activer le fond avec réglages fixes
add_bg_image("fond.png", overlay_opacity=0.90, fixed=True, size="auto")



# ── Fond de plan Esri
ESRI_SATELLITE_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
ESRI_ATTR = "Esri — World Imagery"

# ─────────────────────────────────────────────────────────────
# CONSTANTES MÉTIER
# ─────────────────────────────────────────────────────────────
DEFAULT_WIDTHS: Dict[str, float] = {
    "BAU": 1.0,
    "BDG": 1.0,
    "VL": 3.5,
    "VR": 3.5,
    "VM": 3.5,
    "VS": 3.5,
    "BRET": 3.5,
}
ALL_ELEMENTS = ["BAU", "BDG", "VL", "VR", "VM", "VS", "BRET"]

# Profils (comptages équivalents)
PROFILES: Dict[str, Dict[str, float]] = {
    "2 voies": {"BDG": 1, "VR": 1, "VL": 1, "BAU": 1},
    "2 voies + bretelle": {"BDG": 1, "VR": 1, "VL": 1, "BRET": 1, "BAU": 1},
    "3 voies": {"BDG": 1, "VR": 1, "VM": 1, "VL": 1, "BAU": 1},
    "3 voies + bretelle": {"BDG": 1, "VR": 1, "VM": 1, "VL": 1, "BRET": 1, "BAU": 1},
    "4 voies": {"BDG": 1, "VR": 1, "VM": 1, "VL": 1, "VS": 1, "BAU": 1},
    "4 voies + bretelle": {"BDG": 1, "VR": 1, "VM": 1, "VL": 1, "VS": 1, "BRET": 1, "BAU": 1},
}

# Couleurs profils
PROFILE_COLORS: Dict[str, str] = {
    "2 voies": "#1f77b4",
    "2 voies + bretelle": "#ff7f0e",
    "3 voies": "#2ca02c",
    "3 voies + bretelle": "#9467bd",
    "4 voies": "#d62728",
    "4 voies + bretelle": "#8c564b",
}

# Styles points PR (Fait/Ausculte)
PR_STYLE = {
    ("oui", "oui"): dict(stroke="#27ae60", fill="#FFD700", label="Fait = Oui, Ausculté = Oui"),
    ("oui", "non"): dict(stroke="#27ae60", fill="#FFD700", label="Fait = Oui, Ausculté = Non"),
    ("non", "oui"): dict(stroke="#95a5a6", fill="#0000FF", label="Fait = Non, Ausculté = Oui"),
    ("non", "non"): dict(stroke="#95a5a6", fill="#FFFFFF", label="Fait = Non, Ausculté = Non"),
}

# Matériaux par défaut (densités éditables) pour reprofilage
DEFAULT_MATERIALS = [
    {"matériau": "GB", "densité_t_m3": 2.30, "épaisseur_cm": 0.0},  # Grave-bitume
    {"matériau": "BBTM", "densité_t_m3": 2.40, "épaisseur_cm": 0.0},  # Très mince
    {"matériau": "BBM", "densité_t_m3": 2.35, "épaisseur_cm": 0.0},
    {"matériau": "BBSG", "densité_t_m3": 2.35, "épaisseur_cm": 0.0},
]
if "materials_df" not in st.session_state:
    st.session_state["materials_df"] = pd.DataFrame(DEFAULT_MATERIALS)

# ─────────────────────────────────────────────────────────────
# PROJECTIONS & VILLES
# ─────────────────────────────────────────────────────────────
TO_WGS84 = Transformer.from_crs(2154, 4326, always_xy=True)
TO_L93 = Transformer.from_crs(4326, 2154, always_xy=True)

BOUCHES_DU_RHONE_CITIES = {
    "Marseille": (43.2965, 5.3698),
    "Aix-en-Provence": (43.5297, 5.4474),
    "Arles": (43.6766, 4.6278),
    "Martigues": (43.4058, 5.0480),
    "Salon-de-Provence": (43.6400, 5.0970),
    "Istres": (43.5167, 4.9833),
    "Vitrolles": (43.4600, 5.2489),
    "Miramas": (43.5833, 5.0000),
    "Fos-sur-Mer": (43.5333, 4.9333),
    "Chateauneuf-les-martigues": (43.38835, 5.1492),
    "Aubagne": (43.293046, 5.56842),
}

# ─────────────────────────────────────────────────────────────
# FONCTIONS UTILITAIRES
# ─────────────────────────────────────────────────────────────

def _round_df0(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> pd.DataFrame:
    """Arrondit à 0 décimal toutes les colonnes numériques, sauf celles de `exclude`."""
    df2 = df.copy()
    num_cols = df2.select_dtypes(include=[np.number]).columns.tolist()
    if exclude:
        num_cols = [c for c in num_cols if c not in set(exclude)]
    if num_cols:
        df2[num_cols] = df2[num_cols].round(0)
    return df2


def _normalize_yn(v: str) -> str:
    """Normalise les variantes Oui/Non en {"oui","non"}. Laisse tel quel sinon."""
    s = str(v or "").strip().lower()
    if s in {"oui", "o", "y", "yes", "true", "1"}:
        return "oui"
    if s in {"non", "n", "no", "false", "0"}:
        return "non"
    return s


def planimetric_distance_l93(coords_l93: List[Tuple[float, float]]) -> float:
    """Somme des distances euclidiennes entre points successifs (x,y) en Lambert-93."""
    if len(coords_l93) < 2:
        return 0.0
    return float(
        sum(
            hypot(coords_l93[i + 1][0] - coords_l93[i][0],
                  coords_l93[i + 1][1] - coords_l93[i][1])
            for i in range(len(coords_l93) - 1)
        )
    )


def midpoint_wgs(coords_wgs: List[Tuple[float, float]]) -> Tuple[float, float]:
    lats = [c[0] for c in coords_wgs]
    lons = [c[1] for c in coords_wgs]
    return float(np.mean(lats)), float(np.mean(lons))


def l93_to_wgs(latlon_l93: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """(x,y) -> (lat,lon)"""
    return [TO_WGS84.transform(x, y)[::-1] for x, y in latlon_l93]


def wgs_to_l93(coords_wgs: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """(lat,lon) -> (x,y)"""
    return [TO_L93.transform(lon, lat) for lat, lon in coords_wgs]


def merge_profile_mix(profiles_selected: List[str], percentages: List[float]) -> Dict[str, float]:
    """Fusionne plusieurs profils pondérés (pourcentages normalisés)."""
    weights = np.array(percentages, dtype=float)
    if weights.sum() == 0:
        return {}
    weights = weights / weights.sum()
    agg: Dict[str, float] = {}
    for prof_name, w in zip(profiles_selected, weights):
        prof = PROFILES[prof_name]
        for elem, count in prof.items():
            agg[elem] = agg.get(elem, 0.0) + float(w) * float(count)
    return agg


def dominant_profile_name(profiles_selected: List[str], percentages: List[float]) -> Optional[str]:
    """Profil dominant (max %) ; si égalité ou vide -> None."""
    if not profiles_selected:
        return None
    if len(percentages) != len(profiles_selected):
        return profiles_selected[0]
    if sum(percentages) == 0:
        return None
    max_idx = int(np.argmax(percentages))
    return profiles_selected[max_idx]


def apply_overrides(
    widths: Dict[str, float],
    overrides: Optional[pd.DataFrame],
    route: str,
    cote: str,
    pr_start: float,
    pr_end: float,
) -> Dict[str, float]:
    """
    Applique des surcharges de largeur sur un tronçon (intersection PR).
    - éligible si (ov.pr_start <= pr_end) & (ov.pr_end >= pr_start)
    - si plusieurs lignes ciblent le même élément, la dernière l’emporte.
    """
    if overrides is None or overrides.empty:
        return widths
    mask = (
        (overrides["route"] == route)
        & (overrides["cote"] == cote)
        & (overrides["pr_start"] <= pr_end)
        & (overrides["pr_end"] >= pr_start)
    )
    ov = overrides.loc[mask]
    if ov.empty:
        return widths
    new_widths = widths.copy()
    for _, row in ov.iterrows():
        elem = str(row["element"]) if "element" in row else None
        if elem and elem in new_widths:
            val = float(row["largeur_m"]) if "largeur_m" in row else new_widths[elem]
            new_widths[elem] = val
    return new_widths


def compute_areas(
    distance_m: float,
    widths_m: Dict[str, float],
    element_counts: Dict[str, float],
    included_elements: List[str],
) -> Tuple[pd.DataFrame, float]:
    """Calcule les surfaces par élément : area = distance × largeur équivalente."""
    rows: List[Dict[str, Any]] = []
    total = 0.0
    for elem, count in element_counts.items():
        if elem not in included_elements:
            continue
        width = float(widths_m.get(elem, 0.0))
        width_equiv = float(count) * width
        area = float(distance_m) * width_equiv
        rows.append(
            {
                "element": elem,
                "count_equiv": round(float(count), 3),
                "width_m": round(width, 3),
                "width_equiv_m": round(width_equiv, 3),
                "area_m2": area,
            }
        )
        total += area
    df = pd.DataFrame(rows).sort_values("element").reset_index(drop=True)
    return df, float(total)


def build_pr_popup_html(r: pd.Series) -> str:
    """Construit le contenu HTML du popup PR (structure, badges, V_L/M/R)."""
    route = str(r.get("route", "") or "").strip()
    pr = str(r.get("pr", "") or "").strip()
    cote = str(r.get("cote", "") or "").strip()
    struct = str(r.get("structure") or "Non renseignée").strip()
    fait = _normalize_yn(r.get("Fait", ""))
    ausc = _normalize_yn(r.get("Ausculte", ""))
    refaire = _normalize_yn(r.get("A_refaire", ""))
    v_l = str(r.get("V_L", "") or "").strip()
    v_m = str(r.get("V_M", "") or "").strip()
    v_r = str(r.get("V_R", "") or "").strip()

    def badge(label: str, ok: bool) -> str:
        color_bg = "#eafaf1" if ok else "#fdecea"
        color_fg = "#1e824c" if ok else "#c0392b"
        icon = "✔️" if ok else "❌"
        return (
            f'<span style="display:inline-block;padding:3px 8px;border-radius:12px;'
            f'font-weight:600;font-size:12px;background:{color_bg};color:{color_fg};'
            f'border:1px solid {color_fg}22;margin-right:6px;">{icon} {label}</span>'
        )

    badge_ausc = badge("Ausculte", ausc == "oui")
    badge_fait = badge("Fait", fait == "oui")
    badge_refaire = badge("À refaire", refaire == "oui")

    html = f"""
<div style="font-family: Arial, sans-serif; font-size: 13px; color: #2c3e50; line-height: 1.35;
            background: #ffffff; border-radius: 10px; padding: 10px; box-shadow: 0 6px 20px rgba(0,0,0,0.15);
            min-width: 160px; max-width: 300px;">
  <div style="font-weight: 800; font-size: 10px; margin-bottom: 6px; color:#1f2d3d;">
    PR {pr} — {route} <span style="color:#7f8c8d;">({cote})</span>
  </div>

  <!-- Page 1 -->
  <div id="page1">
    <table style="width:100%; border-collapse: collapse; margin-bottom: 8px;">
      <tr>
        <td style="padding:4px 8px;color:#566573;"><b>Structure</b></td>
        <td style="padding:4px 8px;color:#2c3e50;">{struct}</td>
      </tr>
    </table>
    <div>{badge_ausc}{badge_fait}{badge_refaire}</div>
    <div style="text-align:right;margin-top:8px;">
      <a href="#" onclick="document.getElementById('page1').style.display='none';
                             document.getElementById('page2').style.display='block';return false;"
         style="color:#2980b9;font-size:12px;text-decoration:none;">➡ Structure conseillée</a>
    </div>
  </div>

  <!-- Page 2 -->
  <div id="page2" style="display:none;">
    <div style="font-weight:bold;margin-bottom:6px;">Structure conseillée</div>
    <table style="width:100%; border-collapse: collapse; margin-bottom: 8px;">
      <tr><td style="padding:4px 8px;color:#566573;"><b>V_L</b></td>
          <td style="padding:4px 8px;color:#2c3e50;">{v_l or "—"}</td></tr>
      <tr><td style="padding:4px 8px;color:#566573;"><b>V_M</b></td>
          <td style="padding:4px 8px;color:#2c3e50;">{v_m or "—"}</td></tr>
      <tr><td style="padding:4px 8px;color:#566573;"><b>V_R</b></td>
          <td style="padding:4px 8px;color:#2c3e50;">{v_r or "—"}</td></tr>
    </table>
    <div style="text-align:right;">
      <a href="#" onclick="document.getElementById('page2').style.display='none';
                             document.getElementById('page1').style.display='block';return false;"
         style="color:#2980b9;font-size:12px;text-decoration:none;">⬅ Retour</a>
    </div>
  </div>
</div>
"""
    return html


def make_pr_points_legend(df: pd.DataFrame) -> str:
    """Légende auto des points PR visibles (df déjà filtré)."""
    if df is None or df.empty:
        body = '<div style="color:#999;">Aucun point à afficher</div>'
    else:
        tmp = df.copy()
        tmp["Fait"] = tmp["Fait"].map(_normalize_yn)
        tmp["Ausculte"] = tmp["Ausculte"].map(_normalize_yn)
        counts = tmp.groupby(["Fait", "Ausculte"]).size().to_dict()
        rows = []
        for key, sty in PR_STYLE.items():
            n = counts.get(key, 0)
            if n == 0:
                continue
            border = sty["stroke"]
            fill = sty["fill"]
            label = sty["label"]
            rows.append(f"""
<div style="display:flex;align-items:center;margin:2px 0;">
  <svg width="18" height="18" style="margin-right:8px;">
    <circle cx="9" cy="9" r="6" stroke="{border}" stroke-width="2" fill="{fill}" />
  </svg>
  <div>{label} <span style=\"color:#666\">({n})</span></div>
</div>
""")
        body = "".join(rows) if rows else '<div style="color:#999;">Aucun point après filtrage</div>'

    html = f"""
<div id="pr-maplegend" class="maplegend"
     style="position:absolute; z-index:9999; left:20px; bottom:100px;
            border:2px solid #bbb; background-color:rgba(255,255,255,0.9);
            border-radius:6px; padding:10px; font-size:15px; max-width:280px;">
  <div style="font-weight:700; margin-bottom:6px;">PR – Statuts</div>
  {body}
  <div style="margin-top:6px;color:#666;font-size:11px;">
    Bordure = Fait, Remplissage = Ausculté, Anneau rouge = À refaire
  </div>
</div>
"""
    return html



# ─────────────────────────────────────────────────────────────
# CORRECTION PR_DELTA_m
# ─────────────────────────────────────────────────────────────


def pr_delta_m(pr_start: float, pr_end: float) -> float:
    """
    Retourne la distance en mètres entre deux PR (points de repère).
    Hypothèse : 1 PR = 1000 m.
    """
    try:
        return max((float(pr_end) - float(pr_start)) * 1000.0, 0.0)
    except (TypeError, ValueError):
        return 0.0



# ─────────────────────────────────────────────────────────────
# CHARGEMENT DES FICHIERS
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_pr_file(uploaded_file) -> pd.DataFrame:
    """
    Charge un fichier PR CSV (sep=';', decimal=',') ou Excel.
    Normalise : route, cote, pr, x, y, chainage_m (optionnelle ; 'cumul' mappé).
    """
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, sep=";", decimal=",")
    else:
        xls = pd.ExcelFile(uploaded_file)
        sheet_pr = xls.sheet_names[0]
        df = pd.read_excel(xls, sheet_name=sheet_pr)

    df = df.rename(
        columns={
            "Route": "route", "ROUTE": "route",
            "Cote": "cote", "COTE": "cote",
            "PR": "pr", "Pr": "pr",
            "X": "x", "Y": "y",
            "Chainage_m": "chainage_m", "CHAINAGE_M": "chainage_m",
            "chainage_m": "chainage_m", "cumul": "chainage_m",
        }
    )
    required = ["route", "cote", "pr", "x", "y"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Colonnes manquantes: {missing}. Requis: {required} (+ chainage_m optionnelle)."
        )

    df["route"] = df["route"].astype(str)
    df["cote"] = df["cote"].astype(str)
    df["pr"] = pd.to_numeric(df["pr"], errors="coerce")
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    if "chainage_m" in df.columns:
        df["chainage_m"] = pd.to_numeric(df["chainage_m"], errors="coerce")

    df = df.dropna(subset=["route", "cote", "pr", "x", "y"]).sort_values(["route", "cote", "pr"]).reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def load_overrides_file(uploaded_file) -> Optional[pd.DataFrame]:
    """Charge un fichier d'overrides (CSV/Excel)."""
    if uploaded_file is None:
        return None
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        ov = pd.read_csv(uploaded_file, sep=";", decimal=",")
    else:
        xls = pd.ExcelFile(uploaded_file)
        lower_names = [s.lower() for s in xls.sheet_names]
        if "overrides" in lower_names:
            idx = lower_names.index("overrides")
            sheet_name = xls.sheet_names[idx]
        else:
            sheet_name = xls.sheet_names[0]
        ov = pd.read_excel(xls, sheet_name=sheet_name)

    ov = ov.rename(
        columns={
            "Route": "route", "Cote": "cote",
            "PR_start": "pr_start", "Pr_start": "pr_start",
            "PR_end": "pr_end", "Pr_end": "pr_end",
            "Element": "element", "element": "element",
            "Largeur_m": "largeur_m", "largeur_m": "largeur_m",
        }
    )
    needed_ov = ["route", "cote", "pr_start", "pr_end", "element", "largeur_m"]
    missing_ov = [c for c in needed_ov if c not in ov.columns]
    if missing_ov:
        raise ValueError(f"Overrides: colonnes manquantes {missing_ov}. Attendu: {needed_ov}")

    ov["route"] = ov["route"].astype(str)
    ov["cote"] = ov["cote"].astype(str)
    ov["pr_start"] = pd.to_numeric(ov["pr_start"], errors="coerce")
    ov["pr_end"] = pd.to_numeric(ov["pr_end"], errors="coerce")
    ov["element"] = ov["element"].astype(str)
    ov["largeur_m"] = pd.to_numeric(ov["largeur_m"], errors="coerce")
    ov = ov.dropna()
    return ov


def _prep_dirmed_df(df_src: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Prépare un DataFrame pour la couche DIRMED à partir du df déjà chargé :
    - garde colonnes utiles (avec colonnes optionnelles si présentes),
    - convertit x,y (virgule -> point) en float,
    - projette en WGS84 via TO_WGS84 (EPSG:2154 -> 4326),
    - normalise Fait/Ausculte/A_refaire en minuscule.
    Retourne None si colonnes minimales absentes / invalides.
    """
    needed = {"route", "pr", "x", "y", "cote", "Fait", "Ausculte", "structure"}
    if not needed.issubset(set(df_src.columns)):
        return None
    optional = ["A_refaire", "V_L", "V_M", "V_R"]
    cols = list(needed) + [c for c in optional if c in df_src.columns]
    d = df_src[cols].copy()

    for c in ["x", "y"]:
        d[c] = (
            d[c].astype(str)
            .str.replace("\u00a0", "", regex=False)
            .str.replace(",", ".", regex=False)
            .str.strip()
        )
    d["x"] = pd.to_numeric(d["x"], errors="coerce")
    d["y"] = pd.to_numeric(d["y"], errors="coerce")

    d["route"] = d["route"].astype(str).str.strip()
    d["pr"] = d["pr"].astype(str).str.replace(",", ".", regex=False).str.strip()
    d["cote"] = d["cote"].astype(str).str.strip()

    for c in ["Fait", "Ausculte", "A_refaire"]:
        if c in d.columns:
            d[c] = d[c].astype(str).str.strip().str.lower()

    d = d.dropna(subset=["x", "y"])
    if d.empty:
        return None

    lons, lats = TO_WGS84.transform(d["x"].to_numpy(), d["y"].to_numpy())
    d["lat"] = lats
    d["lon"] = lons
    return d


def parse_drawn_polylines(map_data: Optional[Dict[str, Any]]) -> List[List[Tuple[float, float]]]:
    """
    Extrait toutes les polylignes dessinées (LineString, MultiLineString, FeatureCollection).
    Retourne une liste de polylignes (liste de (lat, lon)).
    """
    if not map_data:
        return []
    polys: List[List[Tuple[float, float]]] = []

    def _extract_from_geom(geom: Dict[str, Any]):
        gtype = (geom or {}).get("type")
        coords = (geom or {}).get("coordinates", [])
        if gtype == "LineString":
            if coords:
                polys.append([(lat, lon) for lon, lat in coords])
        elif gtype == "MultiLineString":
            for line in coords or []:
                if line:
                    polys.append([(lat, lon) for lon, lat in line])
        elif gtype == "FeatureCollection":
            for f in (geom.get("features") or []):
                _extract_from_geom((f or {}).get("geometry") or {})

    last = (map_data or {}).get("last_active_drawing") or {}
    if "geometry" in last:
        _extract_from_geom(last["geometry"])

    drawings = (map_data or {}).get("all_drawings") or []
    for feat in drawings:
        geom = (feat or {}).get("geometry") or {}
        _extract_from_geom(geom)

    return [p for p in polys if p]


def make_legend_html(selected: List[str], percentages: List[int], show_percentages: bool = True) -> str:
    """Légende HTML dynamique des profils avec couleurs et pourcentages."""
    rows = []
    sel_pct = {name: pct for name, pct in zip(selected, percentages)}
    for key, color in PROFILE_COLORS.items():
        label = key.replace("_", " ")
        pct_display = f" — {sel_pct.get(key, 0)}%" if (key in sel_pct and show_percentages) else ""
        weight = "font-weight:600;" if key in sel_pct else "font-weight:400;"
        rows.append(
            f"""
<div style="display:flex;align-items:center;margin:2px 0;{weight}">
  <div style="width:14px;height:14px;background:{color}; border:1px solid #333;margin-right:8px;"></div>
  <div>{label}{pct_display}</div>
</div>
"""
        )
    html = f"""
<div id="maplegend" class="maplegend"
     style="position: absolute; z-index:9999; border:2px solid #bbb;
            background-color: rgba(255, 255, 255, 0.9);
            border-radius:6px; padding:10px; font-size:12px; right: 20px; bottom: 20px;">
  <div class="legend-title" style="font-weight:700; margin-bottom:6px;">Profils & couleurs</div>
  <div class="legend-scale">{''.join(rows)}</div>
  <div style="margin-top:6px;color:#666;">*En gras : profils actuellement sélectionnés pour le prochain dessin.</div>
</div>
"""
    return html


# ─────────────────────────────────────────────────────────────
# UI — Flux unique
# ─────────────────────────────────────────────────────────────
st.title("Estimation de surfaces_Vinci-Construction_SNasri")

# Import des données
with st.container():
    st.markdown("#### Import des données")
    uploaded = st.file_uploader(
        "Fichier PR (CSV ou Excel) : colonnes route, cote, pr, x, y ; optionnelle : chainage_m (dans un CSV, 'cumul' est mappé en chainage_m).",
        type=["xlsx", "csv"],
    )
    col_ov1, col_ov2 = st.columns([1, 2])
    with col_ov1:
        use_overrides = st.checkbox("Ajouter des overrides (optionnel)")
    with col_ov2:
        overrides_file = None
        if use_overrides:
            overrides_file = st.file_uploader(
                "Fichier overrides (CSV/Excel) : route, cote, pr_start, pr_end, element, largeur_m",
                type=["xlsx", "csv"],
                key="ov_file",
            )

if not uploaded:
    st.info("➕ Charge un fichier PR pour commencer.")
    st.stop()

# Lecture des fichiers
try:
    df = load_pr_file(uploaded)
    overrides = load_overrides_file(overrides_file) if overrides_file is not None else None
    st.success(
        f"✅ PR chargés : {len(df)} lignes, {df['route'].nunique()} route(s), {df['cote'].nunique()} côté(s)."
    )
except Exception as e:
    st.error(f"Erreur lors du chargement: {e}")
    st.stop()

# Sélection du segment
st.markdown("---")
st.markdown("#### Sélection du segment")
colA, colB, colC = st.columns(3)
with colA:
    route = st.selectbox("Route", sorted(df["route"].unique()))
with colB:
    cotes_dispo = df.loc[df["route"] == route, "cote"].unique().tolist()
    cote = st.selectbox("Côté", sorted(cotes_dispo))
subset = df[(df["route"] == route) & (df["cote"] == cote)].sort_values("pr").reset_index(drop=True)
with colC:
    prs = sorted(subset["pr"].dropna().unique().tolist())
    pr_start = st.selectbox("PR début", prs, index=0 if prs else None)
    prs_after = [p for p in prs if p > pr_start] if prs else []
    pr_end = st.selectbox("PR fin", prs_after, index=0 if prs_after else None)

sel = subset[subset["pr"].isin([pr_start, pr_end])].sort_values("pr")
if len(sel) != 2:
    st.warning("Sélectionne deux PR valides (début < fin).")
    st.stop()

pr1 = sel.iloc[0]
pr2 = sel.iloc[1]
coords_l93 = [(float(pr1["x"]), float(pr1["y"])), (float(pr2["x"]), float(pr2["y"]))]
coords_wgs = l93_to_wgs(coords_l93)

seg_key = f"{route}__{cote}__{float(pr_start)}->({float(pr_end)})"

# Distance & courbure
st.markdown("---")
st.markdown("#### Distance et courbure")
colD, colE, colF, colF2 = st.columns([1.2, 1, 1, 1.2])
with colD:
    dist_method = st.selectbox(
        "Méthode de distance",
        ["Segment édité", "Chainage", "Droite PR→PR", "PR × 1000 (fixe)", "Fixe"],
        help=(
            "Segment édité = distance de la/les polyligne(s) dessinée(s) sur la carte. "
        ),
    )

default_fixed = pr_delta_m(pr_start, pr_end) or 1000.0
with colE:
    fixed_m = st.number_input("Distance fixe (m)", value=float(default_fixed), step=50.0, min_value=0.0)
with colF:
    curvature_factor = st.number_input("Facteur de courbure", value=1.00, step=0.01, min_value=0.90, max_value=1.20)
with colF2:
    map_height = st.slider("Hauteur carte (px)", min_value=500, max_value=1000, value=750, step=10)
    zoom_init = st.slider("Zoom initial", min_value=10, max_value=18, value=14, step=1)



# Filtrage & légende PR
st.markdown("---")
st.markdown("#### Filtrage - Légende")
filter_options = {
    "Fait Oui / Ausculté Oui": ("oui", "oui"),
    "Fait Oui / Ausculté Non": ("oui", "non"),
    "Fait Non / Ausculté Oui": ("non", "oui"),
    "Fait Non / Ausculté Non": ("non", "non"),
}
col_f1, col_f2 = st.columns([2, 1])
with col_f1:
    selected_filters = st.multiselect(
        "Afficher les statuts PR",
        options=list(filter_options.keys()),
        default=list(filter_options.keys()),
        help="Filtre combiné conforme à la légende PR (bordure = Fait, remplissage = Ausculté).",
    )
with col_f2:
    show_arefaire_only = st.checkbox("Uniquement A_refaire = Oui", value=False)

st.caption("Astuce : décoche des combinaisons pour synchroniser carte et légende.")

dirmed_df_all = _prep_dirmed_df(df)
if dirmed_df_all is not None:
    for c in ["Fait", "Ausculte"]:
        if c in dirmed_df_all.columns:
            dirmed_df_all[c] = dirmed_df_all[c].astype(str).str.strip().str.lower()
else:
    st.info("ℹ️ La couche DIRMED est inactive : colonnes manquantes (Fait, Ausculte, structure) ou coordonnées invalides.")

if dirmed_df_all is None:
    dirmed_df_all = pd.DataFrame(columns=["route", "pr", "cote", "Fait", "Ausculte", "lat", "lon"])  # vide

if selected_filters:
    allowed_pairs = set(filter_options[k] for k in selected_filters)
    mask = dirmed_df_all.apply(lambda r: (r.get("Fait", ""), r.get("Ausculte", "")) in allowed_pairs, axis=1)
    dirmed_df_all = dirmed_df_all[mask]
else:
    dirmed_df_all = dirmed_df_all.iloc[0:0]

if show_arefaire_only:
    if "A_refaire" in dirmed_df_all.columns:
        dirmed_df_all = dirmed_df_all[
            dirmed_df_all["A_refaire"].astype(str).str.strip().str.lower().eq("oui")
        ]
    else:
        st.warning("La colonne 'A_refaire' est absente du fichier : le filtre est ignoré.")

# Largeurs (avec overrides)
st.markdown("---")
with st.expander("⚙️ Largeurs par élément (m) et surcharges par tronçon", expanded=False):
    widths = {
        e: st.number_input(f"{e}", value=float(DEFAULT_WIDTHS.get(e, 0.0)), step=0.1, min_value=0.0)
        for e in ALL_ELEMENTS
    }
    widths_applied = apply_overrides(widths, overrides, route, cote, float(pr_start), float(pr_end))
    if overrides is not None:
        if widths_applied != widths:
            st.info("Des surcharges 'overrides' ont été appliquées à ce tronçon.")
        else:
            st.caption("Aucune surcharge 'overrides' correspondante pour ce tronçon.")
    else:
        widths_applied = widths

# Profils & éléments (avant la carte pour fixer la couleur du dessin)
st.markdown("---")
st.markdown("#### Profils et éléments à inclure")
colG, colH = st.columns([1.1, 1])
with colG:
    if dist_method == "Segment édité":
        profile_simple = st.selectbox(
            "Profil du sous-segment (mode édition simple — pas de pourcentages)",
            list(PROFILES.keys()),
            index=0,
            help="En édition, un seul profil par ligne pour aller plus vite.",
        )
        profiles_selected = [profile_simple]
        percents: List[int] = [100]
        st.caption("Les pourcentages sont masqués en mode 'Segment édité'.")
    else:
        profiles_selected = st.multiselect(
            "Profils à appliquer (pour le prochain dessin ou le segment global)",
            list(PROFILES.keys()),
            default=["2 voies"],
            help="Mix possible via pourcentages (poids normalisés).",
        )
        percents = []
        for name in profiles_selected:
            perc = st.slider(
                f"Part du profil {name.replace('_', ' ')} (%)",
                0, 100, 100 if len(profiles_selected) == 1 else 0, step=5
            )
            percents.append(perc)
    profile_mix = merge_profile_mix(profiles_selected, percents)
    if not profile_mix:
        st.warning("Sélectionne au moins un profil avec une part > 0%.")

dom_name = dominant_profile_name(profiles_selected, percents)
seg_color = PROFILE_COLORS.get(dom_name, "#ff7f50")  # couleur pour nouvelles lignes

with colH:
    preset = st.radio("Préréglages d’inclusion", ["Voies", "Tout", "Personnalisé"], index=1, horizontal=True)
    if preset == "Voies":
        included_elements = [e for e in ["VL", "VR", "VM", "VS"] if e in ALL_ELEMENTS]
    elif preset == "Tout":
        included_elements = ALL_ELEMENTS.copy()
    else:
        included_elements = st.multiselect("Éléments inclus", ALL_ELEMENTS, default=["BDG", "VL", "VR", "VM", "BAU", "BRET"])




# ─────────────────────────────────────────────────────────────
# CARTE INTERACTIVE (ESRI Satellite) & ÉDITION
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### Carte (ESRI Satellite) et édition du tracé")
map_center = midpoint_wgs(coords_wgs)

m = folium.Map(location=map_center, zoom_start=int(zoom_init), control_scale=True, tiles=None)
folium.TileLayer(tiles=ESRI_SATELLITE_URL, attr=ESRI_ATTR, name="Esri Satellite", overlay=False, control=False).add_to(m)

# PR markers
folium.Marker(coords_wgs[0], tooltip=f"PR {int(pr_start)}", icon=folium.Icon(color="green")).add_to(m)
folium.Marker(coords_wgs[1], tooltip=f"PR {int(pr_end)}", icon=folium.Icon(color="red")).add_to(m)

# États en session : géométries éditées & sous-segments
if "edited_geoms" not in st.session_state:
    st.session_state["edited_geoms"] = {}  # {seg_key: List[List[(lat,lon)]]}
if "subsegments" not in st.session_state:
    st.session_state["subsegments"] = {}  # {seg_key: [ {wgs:[...], mix:{...}, color:"#hex", included:[...], profile_name:str, profile_label:str, widths:dict} ]}
if "profile_counts" not in st.session_state:
    st.session_state["profile_counts"] = {}  # {seg_key: {profile_name: count}}

existing_geom = st.session_state["edited_geoms"].get(seg_key, [])

# Afficher sous-segments existants (couleur spécifique)
for it in st.session_state["subsegments"].get(seg_key, []):
    l93_tmp = wgs_to_l93(it["wgs"])
    d_tmp = planimetric_distance_l93(l93_tmp)
    d_tmp = max(d_tmp * float(curvature_factor), 0.0)
    label = it.get("profile_label", "Sous-segment")
    folium.PolyLine(it["wgs"], color=it["color"], weight=6, opacity=0.70, tooltip=f"{label} — {d_tmp:.2f} m").add_to(m)

# Afficher l'existant temporaire (toutes polylignes) en couleur actuelle du profil
if existing_geom:
    if existing_geom and isinstance(existing_geom[0], tuple):
        existing_geom = [existing_geom]
    for line in existing_geom:
        folium.PolyLine(line, color=seg_color, weight=5, opacity=0.95, tooltip="Segment édité").add_to(m)

# Outil de dessin
if dist_method == "Segment édité":
    Draw(
        export=False,
        draw_options={
            "polyline": {"shapeOptions": {"color": seg_color, "weight": 5}},
            "polygon": False,
            "rectangle": False,
            "circle": False,
            "marker": False,
            "circlemarker": False,
        },
        edit_options={"edit": True, "remove": True},
    ).add_to(m)


# Légende profils (masque % en mode édition simple)
legend_html = make_legend_html(profiles_selected, percents, show_percentages=(dist_method != "Segment édité"))
m.get_root()# Légende automatique des points PR (bas-gauche)
pr_legend_html = make_pr_points_legend(dirmed_df_all)
m.get_root()# Couche DIRMED — points & labels de villes
if dirmed_df_all is not None and not dirmed_df_all.empty:
    layer_dirmed = folium.FeatureGroup(name="DIRMED-Auscultations-DTE", show=True)
    for city, (clat, clon) in BOUCHES_DU_RHONE_CITIES.items():
        folium.Marker(
            location=(clat, clon),
            icon=DivIcon(
                icon_size=(120, 10),
                icon_anchor=(0, 0),
                html=f"""
<div style="font-size:12px;font-weight:700;color:#222;
            text-shadow: 0 0 3px #ffffff, 0 0 6px #ffffff; background: rgba(255,255,255,0.0); padding: 0 2px;">
  {city}
</div>""",
            ),
            tooltip=city,
        ).add_to(layer_dirmed)

    for _, r in dirmed_df_all.iterrows():
        fait = _normalize_yn(r.get("Fait", ""))
        ausc = _normalize_yn(r.get("Ausculte", ""))
        aref = _normalize_yn(r.get("A_refaire", ""))
        sty = PR_STYLE.get((fait, ausc), PR_STYLE[("non", "non")])
        tooltip = f"{r.get('route','')} - {r.get('pr','')} ({r.get('cote','')})"
        popup_html = build_pr_popup_html(r)
        iframe = IFrame(html=popup_html, width=320, height=210)
        popup = folium.Popup(iframe, max_width=320)
        folium.CircleMarker(
            location=(float(r["lat"]), float(r["lon"])),
            radius=5,
            color=sty["stroke"],
            weight=2,
            fill=True,
            fill_color=sty["fill"],
            fill_opacity=1.0 if ausc == 'oui' else 0.6,
            tooltip=tooltip,
            popup=popup,
        ).add_to(layer_dirmed)
        if aref == 'oui':
            folium.CircleMarker(
                location=(float(r["lat"]), float(r["lon"])),
                radius=8,
                color="#c0392b",
                weight=2,
                fill=False,
                opacity=0.9,
            ).add_to(layer_dirmed)
    layer_dirmed.add_to(m)
    # ─────────────────────────────────────────────────────────────
# OUTIL : Cercles d'annotation pour PR intermédiaires — Expander repliable
# ─────────────────────────────────────────────────────────────
with st.expander("➕ Ajouter des cercles d'annotation (PR intermédiaires)", expanded=False):
    # État (inchangé)
    if "circles" not in st.session_state:
        st.session_state["circles"] = []

    # Options PR (inchangées)
    pr_options = [f"PR début ({pr_start})", f"PR fin ({pr_end})"] + [f"PR {p}" for p in subset["pr"].tolist()]

    # Champs (compactés mais identiques dans la logique)
    selected_pr = st.selectbox("Choisir le PR de base", pr_options, key="annot_pr_base")
    rayon_m = st.number_input("Rayon (m)", min_value=10.0, step=10.0, value=200.0, key="annot_rayon_m")
    nom_pr = st.text_input("Nom du PR intermédiaire", value=f"{selected_pr} + {int(rayon_m)}", key="annot_nom_pr")

    # Boutons côte à côte pour réduire l'emprise
    col_add, col_del = st.columns(2)
    with col_add:
        add_circle = st.button("➕ Ajouter ce cercle", key="annot_add_circle")
    with col_del:
        clear_circles = st.button("🗑️ Supprimer tous les cercles", key="annot_clear_circles")

    # Actions (logique d'origine inchangée)
    if add_circle:
        if "début" in selected_pr:
            base_point = coords_wgs[0]
        elif "fin" in selected_pr:
            base_point = coords_wgs[1]
        else:
            pr_num = float(selected_pr.replace("PR ", ""))
            row = subset[subset["pr"] == pr_num].iloc[0]
            base_point = TO_WGS84.transform(row["x"], row["y"])[::-1]

        st.session_state["circles"].append(
            {"center": base_point, "radius": rayon_m, "label": nom_pr}
        )

    if clear_circles:
        st.session_state["circles"] = []

# Affichage des cercles sur la carte (indépendant de l'état de l'expander)
for idx, c in enumerate(st.session_state["circles"]):
    folium.Circle(
        location=c["center"], radius=c["radius"],
        color="blue", fill=True, fill_opacity=0.1, tooltip=c["label"]
    ).add_to(m)
    folium.Marker(
        location=c["center"],
        icon=DivIcon(
            icon_size=(150, 36), icon_anchor=(0, 0),
            html=f'<div style="font-size:16px;font-weight:bold;color:#003366;">{c["label"]}</div>'
        ),
    ).add_to(m)


col_map, col_actions = st.columns([4, 1])
with col_map:
    map_data = st_folium(m, height=map_height, width=None) or {}
with col_actions:
    st.markdown("**Actions**")
    if st.button("Réinitialiser le tracé édité"):
        st.session_state["edited_geoms"].pop(seg_key, None)
        st.rerun()

    subsegs = st.session_state["subsegments"].get(seg_key, [])
    if subsegs:
        if st.button("🗑️ Supprimer tous les sous-segments"):
            st.session_state["subsegments"][seg_key] = []
            st.rerun()

    if dist_method == "Segment édité":
        st.caption("🔶 Mode édition actif : dessine/édite une ou plusieurs polylignes.")
        edited_wgs_list = st.session_state["edited_geoms"].get(seg_key) or []
        if edited_wgs_list and isinstance(edited_wgs_list[0], tuple):
            edited_wgs_list = [edited_wgs_list]
        can_add = bool(edited_wgs_list)
        if st.button("➕ Ajouter comme sous-segment", disabled=not can_add):
            chosen = edited_wgs_list[-1]
            dom_for_label = dominant_profile_name(profiles_selected, percents) or "mix"
            counts_map = st.session_state["profile_counts"].get(seg_key, {})
            next_idx = int(counts_map.get(dom_for_label, 0)) + 1
            profile_label = f"{dom_for_label}_{next_idx}"
            counts_map[dom_for_label] = next_idx
            st.session_state["profile_counts"][seg_key] = counts_map
            new_item = {
                "wgs": chosen,
                "mix": merge_profile_mix(profiles_selected, percents),
                "color": PROFILE_COLORS.get(dominant_profile_name(profiles_selected, percents), "#ff7f50"),
                "included": included_elements.copy(),
                "profile_name": dom_for_label,
                "profile_label": profile_label,
                "widths": widths_applied.copy(),
            }
            sub_list = st.session_state["subsegments"].get(seg_key, [])
            sub_list.append(new_item)
            st.session_state["subsegments"][seg_key] = sub_list
            st.session_state["edited_geoms"].pop(seg_key, None)
            st.rerun()
    else:
        st.caption("ℹ️ Mode lecture seule (outils masqués).")

    if st.session_state["subsegments"].get(seg_key):
        st.markdown("**Sous-segments saisis**")
        to_delete = None
        for idx, it in enumerate(st.session_state["subsegments"][seg_key]):
            l93 = wgs_to_l93(it["wgs"])
            dist = planimetric_distance_l93(l93)
            dist = max(dist * float(curvature_factor), 0.0)
            label = it.get("profile_label", f"{it.get('profile_name','mix')}_{idx+1}")
            if "widths" not in it or not isinstance(it["widths"], dict):
                it["widths"] = widths_applied.copy()
            st.write(f"#{idx+1} — {label} — {dist:.2f} m")
            with st.expander(f"Modifier les largeurs pour {label}", expanded=False):
                st.caption("Ces largeurs n'affectent que ce sous‑segment.")
                cols = st.columns(3)
                for j, e in enumerate(ALL_ELEMENTS):
                    with cols[j % 3]:
                        current_val = float(it["widths"].get(e, DEFAULT_WIDTHS.get(e, 0.0)))
                        it["widths"][e] = st.number_input(
                            f"{e}", value=current_val, step=0.1, min_value=0.0, key=f"w_{seg_key}_{idx}_{e}"
                        )
                c1, c2, c3 = st.columns([1, 1, 2])
                with c1:
                    if st.button("↺ Reprendre les largeurs globales", key=f"copy_global_{idx}"):
                        it["widths"] = widths_applied.copy()
                        st.rerun()
                with c2:
                    if st.button("↺ Réinitialiser (valeurs par défaut)", key=f"reset_defaults_{idx}"):
                        it["widths"] = {e: float(DEFAULT_WIDTHS.get(e, 0.0)) for e in ALL_ELEMENTS}
                        st.rerun()
            if st.button(f"Supprimer #{idx+1}", key=f"del_{idx}"):
                to_delete = idx
        if to_delete is not None:
            st.session_state["subsegments"][seg_key].pop(to_delete)
            st.rerun()

# Mise à jour de la géométrie depuis la carte (mode édition)
if dist_method == "Segment édité":
    drawn_list = parse_drawn_polylines(map_data)
    if drawn_list:
        st.session_state["edited_geoms"][seg_key] = drawn_list

# ─────────────────────────────────────────────────────────────
# CALCUL DES DISTANCES & SURFACES
# ─────────────────────────────────────────────────────────────
subsegs = st.session_state["subsegments"].get(seg_key, [])
areas_df = pd.DataFrame([])
total_area = 0.0
distance_display_m = 0.0

if dist_method == "Segment édité" and subsegs:
    rows_all = []
    total_area_all = 0.0
    total_dist_all = 0.0
    for ss_idx, it in enumerate(subsegs, start=1):
        l93 = wgs_to_l93(it["wgs"])
        d = planimetric_distance_l93(l93)
        d = max(d * float(curvature_factor), 0.0)
        widths_this = it.get("widths", widths_applied)
        df_part, area_part = compute_areas(d, widths_this, it["mix"], it["included"])
        df_part["__ss__"] = ss_idx
        df_part["distance_m"] = d
        df_part["profil_nom"] = it.get("profile_label", f"{it.get('profile_name','mix')}_{ss_idx}")
        rows_all.append(df_part)
        total_area_all += area_part
        total_dist_all += d
    areas_df = pd.concat(rows_all, ignore_index=True) if rows_all else pd.DataFrame([])
    total_area = float(total_area_all)
    distance_display_m = float(total_dist_all)
else:
    edited_wgs = st.session_state["edited_geoms"].get(seg_key)
    if edited_wgs and isinstance(edited_wgs[0], tuple):
        edited_wgs = [edited_wgs]
    edited_l93 = [wgs_to_l93(line) for line in edited_wgs] if edited_wgs else None

    # Calcul distance suivant la méthode
    if dist_method == "Segment édité" and edited_l93:
        distance_m = sum(planimetric_distance_l93(line) for line in edited_l93)
    elif dist_method == "Chainage":
        if pd.notna(pr1.get("chainage_m", np.nan)) and pd.notna(pr2.get("chainage_m", np.nan)):
            distance_m = float(pr2["chainage_m"] - pr1["chainage_m"])
        elif pd.notna(pr1["pr"]) and pd.notna(pr2["pr"]):
            distance_m = 1000.0 * float(pr2["pr"] - pr1["pr"])
        else:
            distance_m = planimetric_distance_l93(coords_l93)
    elif dist_method == "Droite PR→PR":
        distance_m = planimetric_distance_l93(coords_l93)
    elif dist_method == "PR × 1000 (fixe)":
        distance_m = pr_delta_m(pr_start, pr_end)
    else:  # Fixe
        distance_m = float(fixed_m)

    distance_m = max(distance_m * float(curvature_factor), 0.0)
    areas_df, total_area = compute_areas(distance_m, widths_applied, profile_mix, included_elements)
    distance_display_m = float(distance_m)
    global_label = (dominant_profile_name(profiles_selected, percents) or "mix") + "_1"
    areas_df["distance_m"] = distance_m
    areas_df["profil_nom"] = global_label

# Traduction FR colonnes pour affichage
areas_df_fr = areas_df.rename(columns={
    "element": "élément",
    "count_equiv": "comptage_équivalent",
    "width_m": "largeur_m",
    "width_equiv_m": "largeur_équivalente_m",
    "area_m2": "surface_m2",
    "__ss__": "sous_segment",
    "distance_m": "distance_m",
    "profil_nom": "profil_nom",
})

cols_order = [
    c for c in ["sous_segment", "profil_nom", "distance_m", "élément",
                "comptage_équivalent", "largeur_m", "largeur_équivalente_m", "surface_m2"]
    if c in areas_df_fr.columns
]
areas_df_fr = areas_df_fr[cols_order + [c for c in areas_df_fr.columns if c not in cols_order]]

# Résultats
st.markdown("---")
st.markdown("#### Résultats")
topA, topB, topC, topD = st.columns(4)
with topA:
    st.metric("Distance (m)", f"{distance_display_m:,.0f}".replace(",", " "))
with topB:
    st.metric("Surface totale (m²)", f"{total_area:,.0f}".replace(",", " "))
with topC:
    st.write("Méthode :", dist_method)
with topD:
    st.write("Profil dominant :", (dominant_profile_name(profiles_selected, percents) or "mix / non défini").replace("_", " "))

st.dataframe(areas_df_fr, width="stretch")

# ✅ Récapitulatif global (voirie & éléments)
st.markdown("#### ✅ Récapitulatif global (voirie & éléments)")
surface_totale_voirie = float(areas_df_fr["surface_m2"].sum()) if not areas_df_fr.empty else 0.0
st.write(f"**Surface totale voirie : {surface_totale_voirie:,.0f} m²**".replace(",", " "))
recap_elements = (
    areas_df_fr.groupby("élément", as_index=False)[["surface_m2"]].sum().sort_values("surface_m2", ascending=False)
    if "élément" in areas_df_fr.columns and not areas_df_fr.empty
    else pd.DataFrame(columns=["élément", "surface_m2"])
)
st.dataframe(recap_elements, width="stretch")

# Récap par sous-segment
if "sous_segment" in areas_df_fr.columns and not areas_df_fr.empty:
    with st.expander("Récapitulatif par sous-segment", expanded=False):
        recap = areas_df_fr.groupby("sous_segment", as_index=False)[["surface_m2"]].sum().sort_values("sous_segment")
        st.dataframe(recap, width="stretch")
        csv_bytes = areas_df_fr.to_csv(index=False).encode("utf-8")
        st.download_button("Télécharger le détail (CSV)", data=csv_bytes, file_name="sous_segments_detail.csv", mime="text/csv")
        csv_bytes2 = recap.to_csv(index=False).encode("utf-8")
        st.download_button("Télécharger le récap par sous-segment (CSV)", data=csv_bytes2, file_name="sous_segments_recap.csv", mime="text/csv")
else:
    csv_bytes = areas_df_fr.to_csv(index=False).encode("utf-8")
    st.download_button("Télécharger le détail (CSV)", data=csv_bytes, file_name="detail.csv", mime="text/csv")
    csv_bytes_el = recap_elements.to_csv(index=False).encode("utf-8")
    st.download_button("Télécharger le récap éléments (CSV)", data=csv_bytes_el, file_name="elements_recap.csv", mime="text/csv")

# ─────────────────────────────────────────────────────────────
# Rabotage & Reprofilage
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## Rabotage & Reprofilage")

st.session_state.setdefault("rabot_epaisseurs", {})
st.session_state.setdefault("materials_df", pd.DataFrame(DEFAULT_MATERIALS))
st.session_state.setdefault("reprof_thk_matrix", pd.DataFrame())


def _ensure_surfaces_source(mode_base: str, recap_elements_df: pd.DataFrame, surface_totale: float) -> pd.DataFrame:
    """
    Retourne un DF des surfaces selon la base :
    - 'Toute la voirie' : 1 ligne synthétique (TOUTE_VOIRIE)
    - 'Par élément' : recap_elements (élément + surface_m2)
    """
    if mode_base == "Toute la voirie":
        return pd.DataFrame([{ "élément": "TOUTE_VOIRIE", "surface_m2": surface_totale }])
    df = recap_elements_df.copy()
    if df.empty:
        return pd.DataFrame([{ "élément": "(aucun)", "surface_m2": 0.0 }])
    return df


def _select_elements_block(source_df: pd.DataFrame, key_prefix: str) -> List[str]:
    """UI commune : multi‑sélection des éléments à inclure dans les calculs."""
    opts = source_df["élément"].astype(str).tolist()
    default = opts
    st.markdown("**Éléments à inclure**")
    sel = st.multiselect("Éléments", opts, default=default, key=f"{key_prefix}_elems")
    return sel


def _export_suffix(route: str, cote: str, pr_start: float, pr_end: float) -> str:
    def _slug(s: str) -> str:
        s = str(s).strip().replace(" ", "_")
        allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        return "".join(ch if ch in allowed else "_" for ch in s)
    try:
        _r = _slug(route); _c = _slug(cote); _a = int(float(pr_start)); _b = int(float(pr_end))
        return f"{_r}_{_c}_PR{_a}-{_b}"
    except Exception:
        return "export"


def _safe_default_thickness(mat: str, materials_df: pd.DataFrame) -> float:
    if materials_df is None or materials_df.empty:
        return 0.0
    ser = materials_df.loc[materials_df["matériau"].astype(str) == str(mat), "épaisseur_cm"]
    if ser.empty:
        return 0.0
    try:
        val = ser.iloc[0]
        return float(val) if pd.notna(val) else 0.0
    except Exception:
        return 0.0


def _safe_default_density(mat: str, materials_df: pd.DataFrame) -> float:
    if materials_df is None or materials_df.empty:
        return 0.0
    ser = materials_df.loc[materials_df["matériau"].astype(str) == str(mat), "densité_t_m3"]
    if ser.empty:
        return 0.0
    try:
        val = ser.iloc[0]
        return float(val) if pd.notna(val) else 0.0
    except Exception:
        return 0.0


tab_rabot, tab_reprof = st.tabs(["Rabotage", "Reprofilage"])

# ───────────────── Rabotage ─────────────────
with tab_rabot:
    st.subheader("Rabotage (multi-hauteurs)")
    base_rabot = st.radio(
        "Base de calcul", ["Toute la voirie", "Par élément"], horizontal=True, key="base_rabot_multi",
        help=("Toute la voirie : un total unique. Par élément : un total par BAU/BDG/VL/VR/VM/VS/BRET."),
    )
    rabot_src = _ensure_surfaces_source(base_rabot, recap_elements, surface_totale_voirie).copy()

    elems_sel_rabot = _select_elements_block(rabot_src, "rabot_multi")
    if elems_sel_rabot:
        rabot_src = rabot_src[rabot_src["élément"].isin(elems_sel_rabot)].copy()

    rows = []
    vol_total_rabot = 0.0
    for _, row in rabot_src.iterrows():
        el = str(row["élément"])
        surf = float(row["surface_m2"]) if pd.notna(row["surface_m2"]) else 0.0
        st.markdown(f"### {el} — {surf:,.0f} m²".replace(",", " "))
        key_prefix = f"{seg_key}__{el}"
        list_key = f"rabot_list_{key_prefix}"
        if list_key not in st.session_state or not isinstance(st.session_state[list_key], list) or len(st.session_state[list_key]) == 0:
            migrated = None
            try:
                if base_rabot == "Par élément" and el != "TOUTE_VOIRIE":
                    prev = st.session_state.get("rabot_epaisseurs", {}).get(el)
                    if prev is not None:
                        migrated = [{"label": "Passe héritée", "h": float(prev)}]
            except Exception:
                migrated = None
            st.session_state[list_key] = migrated or [{"label": "Passe 1", "h": 0.0}]
        passes = st.session_state[list_key]

        with st.expander("Mise à jour rapide des hauteurs pour cet élément", expanded=False):
            c_mass1, c_mass2 = st.columns([2, 1])
            with c_mass1:
                new_h = st.number_input(
                    f"Hauteur commune (cm) pour {el}", min_value=0.0, step=0.5, value=0.0,
                    key=f"rabot_mass_val_{key_prefix}",
                )
            with c_mass2:
                if st.button(f"Appliquer à toutes les passes de {el}", key=f"rabot_mass_apply_{key_prefix}", use_container_width=True):
                    for i in range(len(passes)):
                        passes[i]["h"] = float(new_h)
                    st.rerun()

        for idx, item in enumerate(list(passes)):
            col1, col2, col3 = st.columns([2, 2, 1], vertical_alignment="center")
            with col1:
                label = st.text_input(
                    f"Nom passe {idx+1}", value=str(item.get("label", f"Passe {idx+1}")), key=f"rab_lbl_{key_prefix}_{idx}",
                )
            with col2:
                h = st.number_input(
                    "Hauteur (cm)", min_value=0.0, step=0.5, value=float(item.get("h", 0.0)), key=f"rab_h_{key_prefix}_{idx}",
                )
            with col3:
                if st.button("❌", key=f"rab_del_{key_prefix}_{idx}"):
                    passes.pop(idx)
                    st.rerun()
            passes[idx]["label"], passes[idx]["h"] = label, h

            vol = surf * (h / 100.0)  # m³ = m² × (cm/100)
            rows.append({"élément": el, "passe": label, "surface_m2": surf, "hauteur_cm": h, "volume_m3": vol})
            vol_total_rabot += vol

        c_add, c_reset = st.columns([1, 1])
        with c_add:
            if st.button(f"+ Ajouter une passe pour {el}", key=f"rab_add_{key_prefix}", type="secondary"):
                passes.append({"label": f"Passe {len(passes)+1}", "h": 0.0})
                st.rerun()
        with c_reset:
            if st.button(f"⟲ Réinitialiser {el}", key=f"rab_reset_{key_prefix}"):
                st.session_state[list_key] = [{"label": "Passe 1", "h": 0.0}]
                st.rerun()

    df_rabot = pd.DataFrame(rows)
    st.markdown("### ✅ Détail rabotage (multi-hauteurs)")
    if not df_rabot.empty:
        view_cols = ["élément", "passe", "surface_m2", "hauteur_cm", "volume_m3"]
        st.dataframe(df_rabot[view_cols], width="stretch")
        k1, k2 = st.columns(2)
        with k1:
            st.metric("Surface totale (sélection) m²", f"{df_rabot['surface_m2'].sum():,.0f}".replace(",", " "))
        with k2:
            st.metric("Volume total rabotage (m³)", f"{vol_total_rabot:,.2f}".replace(",", " "))
        st.markdown("#### Totaux par élément")
        tot_el = df_rabot.groupby("élément", as_index=False)[["volume_m3"]].sum().sort_values("volume_m3", ascending=False)
        st.dataframe(tot_el, width="stretch")
        suf = _export_suffix(route, cote, pr_start, pr_end)
        st.download_button(
            "Télécharger le détail (CSV)", data=df_rabot[view_cols].to_csv(index=False).encode("utf-8"),
            file_name=f"rabotage_multi_detail_{suf}.csv", mime="text/csv",
        )
        st.download_button(
            "Télécharger totaux par élément (CSV)", data=tot_el.to_csv(index=False).encode("utf-8"),
            file_name=f"rabotage_multi_totaux_par_element_{suf}.csv", mime="text/csv",
        )
    else:
        st.info("Aucune passe de rabotage saisie pour cette sélection.")

# ───────────────── Reprofilage ─────────────────
with tab_reprof:
    st.subheader("Reprofilage simplifié (multi-matériaux)")
    base_reprof = st.radio("Base de calcul", ["Toute la voirie", "Par élément"], horizontal=True, key="base_reprof_simple")
    reprof_src = _ensure_surfaces_source(base_reprof, recap_elements, surface_totale_voirie).copy()

    materials_df = st.session_state.get("materials_df", pd.DataFrame(DEFAULT_MATERIALS))
    if materials_df.empty:
        st.warning("⚠️ Aucun matériau défini. Ajoute des matériaux avec une densité (t/m³) et une épaisseur par défaut (cm).")
        st.stop()
    mat_opts = materials_df["matériau"].astype(str).tolist()

    rows = []
    for _, row in reprof_src.iterrows():
        el = str(row["élément"])
        surf = float(row["surface_m2"]) if pd.notna(row["surface_m2"]) else 0.0
        st.markdown(f"### {el} — {surf:,.0f} m²".replace(",", " "))
        key_prefix = f"{seg_key}__{el}"
        list_key = f"mats_{key_prefix}"
        if list_key not in st.session_state or not isinstance(st.session_state[list_key], list) or len(st.session_state[list_key]) == 0:
            default_mat = mat_opts[0]
            st.session_state[list_key] = [{"mat": default_mat, "ep": float(_safe_default_thickness(default_mat, materials_df))}]
        mats_list = st.session_state[list_key]

        for idx, item in enumerate(list(mats_list)):
            col1, col2, col3, col4 = st.columns([2, 2, 2, 1], vertical_alignment="center")
            with col1:
                try:
                    sel_idx = mat_opts.index(str(item.get("mat", mat_opts[0])))
                except ValueError:
                    sel_idx = 0
                mat = st.selectbox(f"Matériau {idx+1}", mat_opts, index=sel_idx, key=f"mat_{key_prefix}_{idx}")
            with col2:
                dens = float(_safe_default_density(mat, materials_df))
                st.write(f"Densité : **{dens:.2f} t/m³**")
            with col3:
                default_ep = float(_safe_default_thickness(mat, materials_df))
                ep = st.number_input(f"Épaisseur {idx+1} (cm)", min_value=0.0, step=0.5, value=float(item.get("ep", default_ep)), key=f"ep_{key_prefix}_{idx}")
            with col4:
                if st.button("❌", key=f"del_{key_prefix}_{idx}"):
                    mats_list.pop(idx)
                    st.rerun()
            mats_list[idx]["mat"], mats_list[idx]["ep"] = mat, ep

            vol = surf * (ep / 100.0)
            ton = vol * dens
            rows.append({
                "élément": el, "matériau": mat, "surface_m2": surf, "épaisseur_cm": ep,
                "densité_t_m3": dens, "volume_m3": vol, "tonnage_t": ton,
            })

        c_add, c_reset = st.columns([1, 1])
        with c_add:
            if st.button(f"+ Ajouter un matériau pour {el}", key=f"add_{key_prefix}"):
                default_mat = mat_opts[0]
                mats_list.append({"mat": default_mat, "ep": float(_safe_default_thickness(default_mat, materials_df))})
                st.rerun()
        with c_reset:
            if st.button(f"⟲ Réinitialiser {el}", key=f"reset_{key_prefix}"):
                default_mat = mat_opts[0]
                st.session_state[list_key] = [{"mat": default_mat, "ep": float(_safe_default_thickness(default_mat, materials_df))}]
                st.rerun()

    df_calc = pd.DataFrame(rows)
    if not df_calc.empty:
        view_cols = ["élément", "matériau", "surface_m2", "densité_t_m3", "épaisseur_cm", "volume_m3", "tonnage_t"]
        st.markdown("### ✅ Détail des calculs")
        st.dataframe(df_calc[view_cols], width="stretch")
        vol_total = float(df_calc["volume_m3"].sum())
        ton_total = float(df_calc["tonnage_t"].sum())
        k1, k2 = st.columns(2)
        with k1:
            st.metric("Volume total reprofilage (m³)", f"{vol_total:,.2f}".replace(",", " "))
        with k2:
            st.metric("Tonnage total (t)", f"{ton_total:,.2f}".replace(",", " "))
        st.markdown("#### Totaux par élément")
        tot_el = df_calc.groupby("élément", as_index=False)[["volume_m3", "tonnage_t"]].sum().sort_values("tonnage_t", ascending=False)
        st.dataframe(tot_el, width="stretch")
        st.markdown("#### Totaux par matériau")
        tot_mat = df_calc.groupby("matériau", as_index=False)[["volume_m3", "tonnage_t"]].sum().sort_values("tonnage_t", ascending=False)
        st.dataframe(tot_mat, width="stretch")
        suf = _export_suffix(route, cote, pr_start, pr_end)
        st.download_button("Télécharger le détail (CSV)", data=df_calc[view_cols].to_csv(index=False).encode("utf-8"), file_name=f"reprofilage_detail_{suf}.csv", mime="text/csv")
        st.download_button("Télécharger totaux par élément (CSV)", data=tot_el.to_csv(index=False).encode("utf-8"), file_name=f"reprofilage_totaux_par_element_{suf}.csv", mime="text/csv")
        st.download_button("Télécharger totaux par matériau (CSV)", data=tot_mat.to_csv(index=False).encode("utf-8"), file_name=f"reprofilage_totaux_par_materiau_{suf}.csv", mime="text/csv")
    else:
        st.info("Aucune saisie de matériaux/épaisseurs n’a encore été effectuée pour cette sélection.")

# ─────────────────────────────────────────────────────────────
# Calcul batch (optionnel)
# ─────────────────────────────────────────────────────────────
with st.expander("Calculer tous les intervalles consécutifs de ce côté (batch)", expanded=False):
    run_batch = st.checkbox("Activer le calcul batch")
    if run_batch:
        results = []
        pr_vals = sorted(subset["pr"].dropna().unique().tolist())
        for a, b in zip(pr_vals[:-1], pr_vals[1:]):
            row_a = subset[subset["pr"] == a].iloc[0]
            row_b = subset[subset["pr"] == b].iloc[0]
            if pd.notna(row_a.get("chainage_m", np.nan)) and pd.notna(row_b.get("chainage_m", np.nan)):
                d = float(row_b["chainage_m"] - row_a["chainage_m"])
            else:
                d = planimetric_distance_l93([(float(row_a["x"]), float(row_a["y"])), (float(row_b["x"]), float(row_b["y"]))])
            d = max(d * float(curvature_factor), 0.0)
            widths_pair = apply_overrides(widths_applied, overrides, route, cote, float(a), float(b))
            _, total_pair = compute_areas(d, widths_pair, profile_mix, included_elements)
            results.append({"route": route, "côté": cote, "PR_début": a, "PR_fin": b, "distance_m": d, "surface_totale_m²": total_pair})
        res_df_fr = pd.DataFrame(results)
        st.dataframe(res_df_fr, width="stretch")
        csv_bytes = res_df_fr.to_csv(index=False).encode("utf-8")
        st.download_button("Télécharger le récap batch (CSV)", data=csv_bytes, file_name="batch_surfaces.csv", mime="text/csv")

# ─────────────────────────────────────────────────────────────
# Aide rapide
# ─────────────────────────────────────────────────────────────
with st.expander("Aide rapide", expanded=False):
    st.markdown(
        """
- **Import** : CSV (`;` et `,` décimale) ou Excel. Colonnes requises : `route`, `cote`, `pr`, `x`, `y`.
  Colonne optionnelle : `chainage_m` (dans un CSV, `cumul` → `chainage_m`).
- **Distances** :
  - **Segment édité** : distance planimétrique de la/les polyligne(s) éditée(s).
  - **Chainage** : utilise `chainage_m` si présent, sinon `1000 m` par PR.
  - **Droite PR→PR** : distance droite entre les 2 PR.
  - **PR × 1000 (fixe)** : impose `1000 m` par PR, même si `chainage_m` existe.
  - **Fixe** : valeur imposée (par défaut `1000 × ΔPR`, modifiable).
- **Sous-segments (dessin)** : dessiner une ligne, choisir un profil (unique) et cliquer **“➕ Ajouter comme sous‑segment”**.
  Chaque sous‑segment garde sa couleur (profil dominant) **et affiche sa distance** dans le tableau et sur la carte.
  Les noms sont incrémentés par **profil dominant** (ex. `3 voies_1`, `3 voies_2`, ...).
- **Éléments** : préréglages “Voies”, “Tout”, ou sélection personnalisée.
- **Largeurs** : modifie les largeurs par élément ; **overrides** s’appliquent au **tronçon sélectionné**.
- **Récapitulatif global** : surface totale de la voirie + surface cumulée par élément (tous sous‑segments confondus).
        """
    )
