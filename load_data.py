# transfusion_loaders_with_static_v.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any, Set, Sequence

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split

import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================
# Get the directory where this file is located
_DATA_DIR = os.path.dirname(os.path.abspath(__file__))

INTRA_XA_PATH = os.path.join(_DATA_DIR, "Intra_xa.csv")
POST_Y_PATH   = os.path.join(_DATA_DIR, "Post_y.csv")
PRE_V_PATH    = os.path.join(_DATA_DIR, "Pre_v.csv")
INTRA_V_PATH  = os.path.join(_DATA_DIR, "Intra_v.csv")


# ============================================================
# Column names
# ============================================================
PATIENT_ID_COL = "patient_id"
CASE_ID_COL    = "case_id"
TIME_COL       = "intraoper_time_periods"
ACTION_CUM_COL = "intraoper_pRBC_culVolume"
Y_LABEL_COL    = "AKI_class_occur"


# ============================================================
# Intra x columns (39)
# ============================================================
X_COLS: List[str] = [
    "intraoper_cumulative_HRV",
    "intraoper_cumulative_hypothermia_auc",
    "intraoper_cumulative_hypoxic_auc",
    "intraoper_cumulative_MAP_auc",
    "intraoper_FIO2_ArtBGA",
    "intraoper_Oxygenation_index",
    "intraoper_tHb_value",
    "intraoper_BE_value",
    "intraoper_cHCO3_value",
    "intraoper_CVP_value",
    "intraoper_Ca_value",
    "intraoper_Na_value",
    "intraoper_K_value",
    "intraoper_Glu_value",
    "intraoper_Lac_value",
    "intraoper_PH_value",
    "intraoper_Norepinephrine_max",
    "intraoper_Dopamine_max",
    "intraoper_Dobutamine_max",
    "intraoper_Isoproterenol_max",
    "intraoper_Adrenaline_max",
    "intraoper_Milrinone_max",
    "intraoper_Terlipressin_used",
    "intraoper_Esmolol_used",
    "intraoper_Baquting_value",
    "intraoper_Tranexamic_Acid_value",
    "intraoper_Ketamine_value",
    "intraoper_Heparin_value",
    "intraoper_protamine_value",
    "intraoper_opioids_MME_value",
    "intraoper_GCs_culmulative_value",
    "intraoper_Platelets_culVolume",
    "intraoper_Plasma_culVolume",
    "intraoper_Autoblood_culVolume",
    "CPBresidualBlood_culVolume",
    "intraoper_Inhalation_anesthetics",
    "intraoper_Propofol_used",
    "intraoper_colloid_culVolume",
    "isultrafilter_used",
]


# ============================================================
# Static v columns
# ============================================================
PRE_V_COLS: List[str] = [
    "gender",  # 男/女
    "adm_age",
    "ethnicity",
    "BMI",
    "cardiac_surgery_history_adm",
    "renal_surgery_history_adm",
    "allergy_history_adm",
    "drinking_history_adm",
    "smoking_history_adm",
    "transfusion_history_adm",
    "preoper_DBP",
    "preoper_Heart_Rate",
    "preoper_Pulse",
    "preoper_SBP",
    "preoper_Temperature",
    "preoper_Respiratory_Rate",
    "d_Valve_Dis",
    "d_Rheumatic_HD",
    "d_Congenital_HD",
    "d_Aortic_related_dis",
    "d_Coronary_HD",
    "d_Cardiac_Tumor",
    "d_Cardiomyopathy",
    "d_Infect_Endocarditis",
    "d_Pericardial_Dis",
    "d_Liver_Disease",
    "d_CKD_Status",
    "d_NYHA_Level",
    "d_Cerebrovascular_Events",
    "d_AF_af_Arrhythmia",
    "d_arrhy_avb",
    "d_arrhy_cp_icd_crt",
    "d_auto_immune",
    "d_copd",
    "d_dm",
    "d_hyper_hypo_thyroidism",
    "d_lipn",
    "d_pvd",
    "d_sepsis",
    "d_htn",
    "d_pulmonary_hypertension",
    "preoperLab_NT_proBNP",
    "preoperLab_CRP",
    "preoperLab_Neutrophil_Percentage",
    "preoperLab_eGFR",
    "preoperLab_PT",
    "preoperLab_TBIL",
    "preoperLab_APTT",
    "preoperLab_WBC_Count",
    "preoperLab_ALB",
    "preoperLab_RBC_Count",
    "preoperLab_TnT",
    "preoperLab_Cystatin_C",
    "preoperLab_Platelet_Count",
    "preoperLab_BUN",
    "preoperLab_ESR",
    "preoperLab_Hemoglobin",
    "preoperLab_Glucose",
    "preoper_LVEF",
    "preoperDrug_Glucocorticoid_Usage",
    "preoperDrug_Amphotericin_Aminoglycoside_Combined_Usage",
    "preoperDrug_ACEIARB_Usage",
    "preoperDrug_Diuretics_Usage",
    "preoperDrug_NSAIDs_Usage",
    "preoperDrug_Norepinephrine",
    "preoperDrug_vasopressinUSE",
    "preoperDrug_Dopamine",
    "preoperDrug_Nitroprusside",
    "preoperDrug_Dobutamine",
    "preoperDrug_Isoproterenol",
    "preoperDrug_Epinephrine",
    "preoperDrug_Nitroglycerin",
    "preoperDrug_Metaraminol",
    "preoperDrug_Contrast_Exposure",
    "pre_aki",
]

INTRA_V_COLS: List[str] = [
    "is_emergency",  # 1 non-emergency, 2 emergency (binary)
    "ASA_class",     # 1..5 (we support ordinal or onehot)
    "CABG_oper",
    "CardiacTumor_oper",
    "HeartTransplant_oper",
    "aortic_oper",
    "congenital_oper",
    "valve_oper",
]


# ============================================================
# ID cleaning
# ============================================================
def _clean_id_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    s = s.replace({"nan": "", "None": "", "<NA>": ""})
    return s

def normalize_patient_id(s: pd.Series, width: Optional[int] = None) -> pd.Series:
    s = _clean_id_series(s)
    if width is None:
        return s
    is_digits = s.str.fullmatch(r"\d+").fillna(False)
    s.loc[is_digits] = s.loc[is_digits].str.zfill(width)
    return s

def normalize_case_id(s: pd.Series) -> pd.Series:
    return _clean_id_series(s)


# ============================================================
# Action bins
# ============================================================
_ACTION_BINS = np.array([0, 1, 2, 3, 4, 5], dtype=np.float32)

def bin_incremental_action(delta: np.ndarray) -> np.ndarray:
    delta = np.asarray(delta, dtype=np.float32)
    delta = np.clip(delta, 0.0, np.inf)
    return np.digitize(delta, bins=_ACTION_BINS, right=True).astype(np.int64)


# ============================================================
# Collate: now returns x, a, y, v, lengths, mask
# ============================================================
def pad_collate_varlen(batch, pad_action: int = 0):
    xs, aas, ys, vs, lens = zip(*batch)
    lengths = torch.stack(lens, dim=0)
    B = len(batch)
    T_max = int(lengths.max().item())
    x_dim = xs[0].shape[-1]
    v_dim = vs[0].shape[-1]

    x_pad = torch.zeros(B, T_max, x_dim, dtype=torch.float32)
    a_pad = torch.full((B, T_max), fill_value=int(pad_action), dtype=torch.long)

    # FIX: y is (B,1) to match the simulator + your model assumptions
    y = torch.stack(ys, dim=0).to(dtype=torch.float32).view(B, 1)

    v = torch.stack(vs, dim=0).to(dtype=torch.float32).view(B, v_dim)

    for i, (x_i, a_i, L) in enumerate(zip(xs, aas, lengths)):
        L = int(L.item())
        x_pad[i, :L, :] = x_i
        a_pad[i, :L] = a_i

    t = torch.arange(T_max).unsqueeze(0).expand(B, T_max)
    mask = t < lengths.unsqueeze(1)
    return x_pad, a_pad, y, v, lengths, mask


# ============================================================
# Diagnostics helpers (reuse from your current code)
# ============================================================
def infer_cumulative_cols(x_cols: List[str]) -> Set[str]:
    keys = ["cumulative", "culmulative", "culvolume", "_auc", "auc"]
    out = set()
    for c in x_cols:
        cl = c.lower()
        if any(k in cl for k in keys):
            out.add(c)
    return out

def diff_within_surgery(df: pd.DataFrame, col: str) -> pd.Series:
    d = df.groupby([PATIENT_ID_COL, CASE_ID_COL], sort=False)[col].diff()
    d = d.fillna(df[col])
    d = d.clip(lower=0.0)
    return d.astype(np.float32)

def detect_binary_cols(df: pd.DataFrame, cols: List[str]) -> Set[str]:
    binary = set()
    for c in cols:
        if c not in df.columns:
            continue
        v = df[c].dropna()
        if v.empty:
            continue
        uniq = np.unique(v.to_numpy(dtype=np.float32))
        if len(uniq) <= 2 and set(uniq.tolist()).issubset({0.0, 1.0}):
            binary.add(c)
    return binary

def list_negative_cols(df: pd.DataFrame, cols: List[str]) -> List[str]:
    neg = []
    for c in cols:
        if c not in df.columns:
            continue
        try:
            if float(df[c].min()) < 0:
                neg.append(c)
        except Exception:
            pass
    return sorted(neg)

def alignment_diagnostics(intra_keys: Set[Tuple[str, str]], post_keys: Set[Tuple[str, str]], sample_n: int = 10) -> Dict[str, Any]:
    inter = intra_keys & post_keys
    only_in_intra = list(intra_keys - post_keys)
    only_in_post  = list(post_keys - intra_keys)
    return {
        "intra_unique_keys": len(intra_keys),
        "post_unique_keys": len(post_keys),
        "intersection_keys": len(inter),
        "missing_in_post_y_count": len(only_in_intra),
        "missing_in_intra_count": len(only_in_post),
        "missing_in_post_y_sample": only_in_intra[:sample_n],
        "missing_in_intra_sample": only_in_post[:sample_n],
    }


# ============================================================
# Post_y loader + diagnostics
# ============================================================
def load_post_y_with_diagnostics(post_y_path: str, pid_width: Optional[int] = None) -> Tuple[Dict[Tuple[str, str], int], Dict[str, Any]]:
    dfy = pd.read_csv(
        post_y_path,
        dtype={PATIENT_ID_COL: "string", CASE_ID_COL: "string"},
        low_memory=False
    )

    dfy[PATIENT_ID_COL] = normalize_patient_id(dfy[PATIENT_ID_COL], width=pid_width)
    dfy[CASE_ID_COL]    = normalize_case_id(dfy[CASE_ID_COL])

    dfy[Y_LABEL_COL] = pd.to_numeric(dfy[Y_LABEL_COL], errors="coerce")
    dfy = dfy.dropna(subset=[Y_LABEL_COL])
    dfy = dfy[dfy[Y_LABEL_COL].isin([0, 1])]

    dup_counts = dfy.groupby([PATIENT_ID_COL, CASE_ID_COL]).size()
    dup_keys = dup_counts[dup_counts > 1].index.tolist()

    conflict_keys = []
    if dup_keys:
        nunique = dfy.groupby([PATIENT_ID_COL, CASE_ID_COL])[Y_LABEL_COL].nunique()
        conflict_keys = nunique[nunique > 1].index.tolist()

    dfy = dfy.drop_duplicates(subset=[PATIENT_ID_COL, CASE_ID_COL], keep="last")

    label_map: Dict[Tuple[str, str], int] = {}
    for _, r in dfy.iterrows():
        key = (str(r[PATIENT_ID_COL]), str(r[CASE_ID_COL]))
        label_map[key] = int(r[Y_LABEL_COL])

    diag = {
        "post_y_rows": int(len(dfy)),
        "post_y_unique_keys": int(len(label_map)),
        "post_y_duplicate_keys_count": int(len(dup_keys)),
        "post_y_conflicting_label_keys_count": int(len(conflict_keys)),
        "post_y_conflicting_label_sample": conflict_keys[:10],
    }
    return label_map, diag


# ============================================================
# Static v loaders
#   - ethnicity: one-hot (train-fitted) + UNK bucket
#   - ASA_class: default one-hot (1..5) + UNK, or ordinal
# ============================================================
def _safe_read_csv(path: str, encoding: Optional[str] = None, **kwargs) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding=encoding, **kwargs)
    except UnicodeDecodeError:
        # fallback (common in csv exports)
        return pd.read_csv(path, encoding="utf-8-sig", **kwargs)

def load_pre_v_df(pre_v_path: str, pid_width: Optional[int] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = _safe_read_csv(
        pre_v_path,
        encoding="utf-8",
        dtype={PATIENT_ID_COL: "string", CASE_ID_COL: "string"},
        low_memory=False
    )

    df[PATIENT_ID_COL] = normalize_patient_id(df[PATIENT_ID_COL], width=pid_width)
    df[CASE_ID_COL]    = normalize_case_id(df[CASE_ID_COL])

    missing = [c for c in PRE_V_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in Pre_v.csv: {missing}")

    keep = [PATIENT_ID_COL, CASE_ID_COL] + PRE_V_COLS
    df = df[keep].copy()

    # gender: 男=1, 女=0
    g = df["gender"].astype(str).str.strip()
    df["gender"] = np.where(g == "男", 1.0, np.where(g == "女", 0.0, np.nan)).astype(np.float32)

    # ethnicity: keep as string (encoded later)
    df["ethnicity"] = df["ethnicity"].astype(str).str.strip().replace({"nan": "", "None": "", "<NA>": ""})

    # numeric conversions for the rest (leave ethnicity)
    numeric_cols = [c for c in PRE_V_COLS if c not in ["gender", "ethnicity"]]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[numeric_cols] = df[numeric_cols].astype(np.float32)

    # duplicates diagnostics
    dup_counts = df.groupby([PATIENT_ID_COL, CASE_ID_COL]).size()
    dup_keys = dup_counts[dup_counts > 1].index.tolist()
    df = df.drop_duplicates(subset=[PATIENT_ID_COL, CASE_ID_COL], keep="last")

    diag = {
        "pre_v_rows": int(len(df)),
        "pre_v_unique_keys": int(df[[PATIENT_ID_COL, CASE_ID_COL]].drop_duplicates().shape[0]),
        "pre_v_duplicate_keys_count": int(len(dup_keys)),
        "pre_v_duplicate_keys_sample": dup_keys[:10],
    }
    return df, diag

def load_intra_v_df(intra_v_path: str, pid_width: Optional[int] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = pd.read_csv(
        intra_v_path,
        dtype={PATIENT_ID_COL: "string", CASE_ID_COL: "string"},
        low_memory=False
    )
    df[PATIENT_ID_COL] = normalize_patient_id(df[PATIENT_ID_COL], width=pid_width)
    df[CASE_ID_COL]    = normalize_case_id(df[CASE_ID_COL])

    missing = [c for c in INTRA_V_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in Intra_v.csv: {missing}")

    keep = [PATIENT_ID_COL, CASE_ID_COL] + INTRA_V_COLS
    df = df[keep].copy()

    # is_emergency: 1 non-emergency, 2 emergency -> binary (emergency=1)
    em = pd.to_numeric(df["is_emergency"], errors="coerce")
    df["is_emergency"] = np.where(em == 2, 1.0, np.where(em == 1, 0.0, np.nan)).astype(np.float32)

    # ASA_class keep numeric for now (encoded later)
    df["ASA_class"] = pd.to_numeric(df["ASA_class"], errors="coerce")

    # other ops -> numeric
    op_cols = [c for c in INTRA_V_COLS if c not in ["is_emergency", "ASA_class"]]
    for c in op_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df[op_cols] = df[op_cols].astype(np.float32)

    dup_counts = df.groupby([PATIENT_ID_COL, CASE_ID_COL]).size()
    dup_keys = dup_counts[dup_counts > 1].index.tolist()
    df = df.drop_duplicates(subset=[PATIENT_ID_COL, CASE_ID_COL], keep="last")

    diag = {
        "intra_v_rows": int(len(df)),
        "intra_v_unique_keys": int(df[[PATIENT_ID_COL, CASE_ID_COL]].drop_duplicates().shape[0]),
        "intra_v_duplicate_keys_count": int(len(dup_keys)),
        "intra_v_duplicate_keys_sample": dup_keys[:10],
    }
    return df, diag


def _index_by_key(df: pd.DataFrame) -> pd.DataFrame:
    return df.set_index([PATIENT_ID_COL, CASE_ID_COL], drop=False)

@dataclass
class StaticEncoders:
    ethnicity_categories: List[str]        # train-fitted, excludes "" (empty); UNK handled separately
    asa_encoding: str                     # "onehot" or "ordinal"
    asa_onehot_classes: List[int]         # [1,2,3,4,5] for onehot
    v_col_names: List[str]                # final v dimension names (for debugging)

def _fit_static_encoders(
    train_keys: Sequence[Tuple[str, str]],
    pre_df: pd.DataFrame,
    intra_df: pd.DataFrame,
    asa_encoding: str = "onehot",
) -> StaticEncoders:
    pre_i = _index_by_key(pre_df)
    intra_i = _index_by_key(intra_df)

    # ethnicity categories from TRAIN only (plus UNK)
    eth_vals = []
    for k in train_keys:
        if k in pre_i.index:
            v = str(pre_i.loc[k, "ethnicity"])
            if v and v not in ("nan", "None", "<NA>"):
                eth_vals.append(v)
    eth_cats = sorted(list(set(eth_vals)))

    # ASA classes: fixed 1..5 for onehot; ordinal uses numeric
    asa_classes = [1, 2, 3, 4, 5]

    # Build final v_col_names
    # Pre numeric columns: all PRE_V_COLS except ethnicity (gender included here as numeric/binary)
    pre_numeric = [c for c in PRE_V_COLS if c != "ethnicity"]
    # We keep ethnicity as onehot in final vector:
    eth_onehot_names = [f"ethnicity__{c}" for c in eth_cats] + ["ethnicity__UNK"]

    # Intra numeric (excluding ASA if onehot)
    intra_numeric = [c for c in INTRA_V_COLS if c != "ASA_class"]

    if asa_encoding == "onehot":
        asa_names = [f"ASA_class__{c}" for c in asa_classes] + ["ASA_class__UNK"]
        v_cols = pre_numeric + eth_onehot_names + intra_numeric + asa_names
    elif asa_encoding == "ordinal":
        v_cols = pre_numeric + eth_onehot_names + intra_numeric + ["ASA_class"]
    else:
        raise ValueError("asa_encoding must be 'onehot' or 'ordinal'")

    return StaticEncoders(
        ethnicity_categories=eth_cats,
        asa_encoding=asa_encoding,
        asa_onehot_classes=asa_classes,
        v_col_names=v_cols,
    )


def _encode_onehot(value: str, categories: List[str]) -> np.ndarray:
    out = np.zeros((len(categories) + 1,), dtype=np.float32)  # + UNK
    if value in categories:
        out[categories.index(value)] = 1.0
    else:
        out[-1] = 1.0
    return out

def _encode_asa(value: Any, enc: StaticEncoders) -> np.ndarray:
    if enc.asa_encoding == "ordinal":
        v = np.float32(np.nan if value is None else float(value))
        if np.isnan(v):
            v = np.float32(0.0)
        return np.array([v], dtype=np.float32)

    # onehot
    out = np.zeros((len(enc.asa_onehot_classes) + 1,), dtype=np.float32)  # + UNK
    try:
        iv = int(float(value))
    except Exception:
        iv = None
    if iv in enc.asa_onehot_classes:
        out[enc.asa_onehot_classes.index(iv)] = 1.0
    else:
        out[-1] = 1.0
    return out

def build_v_for_key(
    key: Tuple[str, str],
    pre_i: pd.DataFrame,
    intra_i: pd.DataFrame,
    enc: StaticEncoders,
    fill_missing_numeric: float = 0.0,
) -> np.ndarray:
    # Pre numeric (all except ethnicity)
    pre_numeric_cols = [c for c in PRE_V_COLS if c != "ethnicity"]
    if key in pre_i.index:
        pre_row = pre_i.loc[key]
        pre_num = np.array([pre_row.get(c, np.nan) for c in pre_numeric_cols], dtype=np.float32)
        eth_val = str(pre_row.get("ethnicity", "")).strip()
    else:
        pre_num = np.full((len(pre_numeric_cols),), np.nan, dtype=np.float32)
        eth_val = ""

    pre_num = np.nan_to_num(pre_num, nan=float(fill_missing_numeric)).astype(np.float32)
    eth_oh = _encode_onehot(eth_val, enc.ethnicity_categories)

    # Intra numeric (excluding ASA)
    intra_numeric_cols = [c for c in INTRA_V_COLS if c != "ASA_class"]
    if key in intra_i.index:
        intra_row = intra_i.loc[key]
        intra_num = np.array([intra_row.get(c, np.nan) for c in intra_numeric_cols], dtype=np.float32)
        asa_val = intra_row.get("ASA_class", np.nan)
    else:
        intra_num = np.full((len(intra_numeric_cols),), np.nan, dtype=np.float32)
        asa_val = np.nan

    intra_num = np.nan_to_num(intra_num, nan=float(fill_missing_numeric)).astype(np.float32)
    asa_enc = _encode_asa(asa_val, enc)

    return np.concatenate([pre_num, eth_oh, intra_num, asa_enc], axis=0).astype(np.float32)


# ============================================================
# Intra_xa: build sequences (now includes v placeholder)
# ============================================================
@dataclass
class SurgerySequence:
    x: np.ndarray
    a: np.ndarray
    y: int
    v: np.ndarray
    length: int
    key: Tuple[str, str]

class SurgeryDataset(Dataset):
    def __init__(self, sequences: List[SurgerySequence]):
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        s = self.sequences[idx]
        return (
            torch.tensor(s.x, dtype=torch.float32),
            torch.tensor(s.a, dtype=torch.long),
            torch.tensor(float(s.y), dtype=torch.float32),
            torch.tensor(s.v, dtype=torch.float32),
            torch.tensor(s.length, dtype=torch.long),
        )


def build_sequences_from_intra_with_diagnostics(
    intra_path: str,
    label_map: Dict[Tuple[str, str], int],
    pre_df: pd.DataFrame,
    intra_v_df: pd.DataFrame,
    static_enc: StaticEncoders,
    use_incremental_for_cumulative_x: bool = True,
    pid_width: Optional[int] = None,
    drop_unlabeled: bool = True,
) -> Tuple[List[SurgerySequence], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    df = pd.read_csv(
        intra_path,
        dtype={PATIENT_ID_COL: "string", CASE_ID_COL: "string"},
        low_memory=False
    )

    df[PATIENT_ID_COL] = normalize_patient_id(df[PATIENT_ID_COL], width=pid_width)
    df[CASE_ID_COL]    = normalize_case_id(df[CASE_ID_COL])

    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL])
    df[TIME_COL] = df[TIME_COL].astype(int)

    df[ACTION_CUM_COL] = pd.to_numeric(df[ACTION_CUM_COL], errors="coerce").fillna(0.0).astype(np.float32)

    missing_x = [c for c in X_COLS if c not in df.columns]
    if missing_x:
        raise ValueError(f"Missing x cols in Intra_xa.csv: {missing_x}")

    for c in X_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[X_COLS] = df[X_COLS].fillna(0.0).astype(np.float32)

    df = df.sort_values([PATIENT_ID_COL, CASE_ID_COL, TIME_COL]).reset_index(drop=True)

    # action incremental -> category
    a_delta = df.groupby([PATIENT_ID_COL, CASE_ID_COL], sort=False)[ACTION_CUM_COL].diff()
    a_delta = a_delta.fillna(df[ACTION_CUM_COL]).astype(np.float32).clip(lower=0.0)
    df["_a_cat"] = bin_incremental_action(a_delta.to_numpy())

    # cumulative x option
    cumulative_cols = infer_cumulative_cols(X_COLS)
    negative_raw = list_negative_cols(df, X_COLS)

    if use_incremental_for_cumulative_x and cumulative_cols:
        for c in cumulative_cols:
            df[c] = diff_within_surgery(df, c)

    negative_post = list_negative_cols(df, X_COLS)
    binary_x_cols = detect_binary_cols(df, X_COLS)

    # static indices
    pre_i = _index_by_key(pre_df)
    intra_i = _index_by_key(intra_v_df)

    # build sequences aligned
    sequences: List[SurgerySequence] = []
    grouped = df.groupby([PATIENT_ID_COL, CASE_ID_COL], sort=False)

    intra_keys = set(grouped.groups.keys())
    post_keys = set(label_map.keys())

    kept = 0
    dropped = 0
    missing_pre = 0
    missing_intra_v = 0

    for key, g in grouped:
        key = (str(key[0]), str(key[1]))

        if key not in label_map:
            if drop_unlabeled:
                dropped += 1
                continue
            y = 0
        else:
            y = int(label_map[key])

        if key not in pre_i.index:
            missing_pre += 1
        if key not in intra_i.index:
            missing_intra_v += 1

        g_sorted = g.sort_values(TIME_COL)
        x = g_sorted[X_COLS].to_numpy(dtype=np.float32)
        a = g_sorted["_a_cat"].to_numpy(dtype=np.int64)
        T = int(len(g_sorted))

        v = build_v_for_key(key, pre_i, intra_i, static_enc, fill_missing_numeric=0.0)

        sequences.append(SurgerySequence(x=x, a=a, y=y, v=v, length=T, key=key))
        kept += 1

    intra_diag = {
        "intra_rows": int(len(df)),
        "intra_unique_keys": int(len(intra_keys)),
        "total_surgeries_in_intra": int(len(intra_keys)),
        "kept_aligned": int(kept),
        "dropped_unlabeled": int(dropped),
        "cumulative_like_cols": sorted(list(cumulative_cols)),
        "binary_cols_detected": sorted(list(binary_x_cols)),
        "negative_x_cols_raw": negative_raw,
        "negative_x_cols_post": negative_post,
    }

    align_diag = alignment_diagnostics(
        intra_keys=set(map(lambda t: (str(t[0]), str(t[1])), intra_keys)),
        post_keys=set(map(lambda t: (str(t[0]), str(t[1])), post_keys)),
        sample_n=10,
    )

    static_diag = {
        "static_dim_v": int(len(static_enc.v_col_names)),
        "missing_pre_v_count_in_intra": int(missing_pre),
        "missing_intra_v_count_in_intra": int(missing_intra_v),
        "pre_v_unique_keys": int(pre_df[[PATIENT_ID_COL, CASE_ID_COL]].drop_duplicates().shape[0]),
        "intra_v_unique_keys": int(intra_v_df[[PATIENT_ID_COL, CASE_ID_COL]].drop_duplicates().shape[0]),
    }

    return sequences, intra_diag, align_diag, static_diag


# ============================================================
# Normalization plans
#   - same rule as x:
#       binary/onehot -> none
#       nonneg -> scale-only
#       else -> zscore
# ============================================================
@dataclass
class NormalizationPlan:
    methods: List[str]       # "none" | "scale" | "zscore"
    mean: np.ndarray
    std: np.ndarray
    scale_cols: List[str]
    zscore_cols: List[str]
    none_cols: List[str]

def make_normalization_plan_from_matrix(
    X_train: np.ndarray,
    col_names: List[str],
    binary_cols: Set[str],
    force_scale_cols: Set[str] | None = None,
    scale_nonneg_only: bool = True,
) -> NormalizationPlan:
    if force_scale_cols is None:
        force_scale_cols = set()

    mean = X_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std  = X_train.std(axis=0, dtype=np.float64).astype(np.float32)
    std  = np.where(std < 1e-8, 1.0, std).astype(np.float32)
    mins = X_train.min(axis=0)

    methods: List[str] = []
    scale_cols, zscore_cols, none_cols = [], [], []

    for j, c in enumerate(col_names):
        if c in binary_cols:
            methods.append("none")
            none_cols.append(c)
            continue

        if c in force_scale_cols:
            methods.append("scale")
            scale_cols.append(c)
            continue

        if scale_nonneg_only and mins[j] >= 0.0:
            methods.append("scale")
            scale_cols.append(c)
        else:
            methods.append("zscore")
            zscore_cols.append(c)

    return NormalizationPlan(
        methods=methods, mean=mean, std=std,
        scale_cols=scale_cols, zscore_cols=zscore_cols, none_cols=none_cols
    )

def apply_plan_inplace_matrix(X: np.ndarray, plan: NormalizationPlan) -> np.ndarray:
    X = X.astype(np.float32, copy=True)
    for j, method in enumerate(plan.methods):
        if method == "none":
            continue
        elif method == "scale":
            X[:, j] = X[:, j] / plan.std[j]
        elif method == "zscore":
            X[:, j] = (X[:, j] - plan.mean[j]) / plan.std[j]
        else:
            raise ValueError(method)
    return X

def apply_normalization_inplace_sequences_x(sequences: List[SurgerySequence], plan: NormalizationPlan) -> None:
    for s in sequences:
        # s.x is (T, dim_x)
        X = s.x.astype(np.float32, copy=True)
        for j, method in enumerate(plan.methods):
            if method == "none":
                continue
            elif method == "scale":
                X[:, j] = X[:, j] / plan.std[j]
            elif method == "zscore":
                X[:, j] = (X[:, j] - plan.mean[j]) / plan.std[j]
            else:
                raise ValueError(method)
        s.x = X

def apply_normalization_inplace_sequences_v(sequences: List[SurgerySequence], plan: NormalizationPlan) -> None:
    for s in sequences:
        s.v = apply_plan_inplace_matrix(s.v.reshape(1, -1), plan).reshape(-1).astype(np.float32)


# ============================================================
# Main entry
# ============================================================
def create_loaders(
    use_incremental_for_cumulative_x: bool = True,
    pid_width: Optional[int] = None,
    train_frac: float = 0.8,
    seed: int = 0,
    batch_size_train: int = 32,
    batch_size_test: int = 64,
    normalize_x: bool = True,
    normalize_v: bool = True,
    scale_nonneg_only: bool = True,
    plot_length_hist: bool = True,
    asa_encoding: str = "onehot",   # "onehot" (recommended) or "ordinal"
):
    # 1) y labels
    label_map, post_diag = load_post_y_with_diagnostics(POST_Y_PATH, pid_width=pid_width)

    # 2) static dfs
    pre_df, pre_diag = load_pre_v_df(PRE_V_PATH, pid_width=pid_width)
    intra_v_df, intra_v_diag = load_intra_v_df(INTRA_V_PATH, pid_width=pid_width)

    # 3) We will split later, but we need encoders fitted on TRAIN keys.
    #    So first: load Intra_xa keys only (cheap pass) to build candidate key list.
    #    We will actually build sequences AFTER fitting encoders, but fitting needs train keys.
    #    To avoid extra passes, we read Intra_xa minimal columns to get key list.
    tmp = pd.read_csv(
        INTRA_XA_PATH,
        dtype={PATIENT_ID_COL: "string", CASE_ID_COL: "string"},
        usecols=[PATIENT_ID_COL, CASE_ID_COL, TIME_COL],
        low_memory=False
    )
    tmp[PATIENT_ID_COL] = normalize_patient_id(tmp[PATIENT_ID_COL], width=pid_width)
    tmp[CASE_ID_COL]    = normalize_case_id(tmp[CASE_ID_COL])
    intra_keys_all = tmp[[PATIENT_ID_COL, CASE_ID_COL]].drop_duplicates()
    all_keys = [(str(r[PATIENT_ID_COL]), str(r[CASE_ID_COL])) for _, r in intra_keys_all.iterrows()]

    # For train/test split reproducibility
    rng = np.random.RandomState(int(seed))
    perm = rng.permutation(len(all_keys))
    n_train_keys = int(round(train_frac * len(all_keys)))
    n_train_keys = max(1, min(n_train_keys, len(all_keys) - 1))
    train_keys = [all_keys[i] for i in perm[:n_train_keys]]
    # Fit encoders on TRAIN keys only
    static_enc = _fit_static_encoders(train_keys, pre_df, intra_v_df, asa_encoding=asa_encoding)

    # 4) build sequences (x,a,y,v) + diagnostics
    sequences, intra_diag, align_diag, static_diag = build_sequences_from_intra_with_diagnostics(
        INTRA_XA_PATH,
        label_map,
        pre_df=pre_df,
        intra_v_df=intra_v_df,
        static_enc=static_enc,
        use_incremental_for_cumulative_x=use_incremental_for_cumulative_x,
        pid_width=pid_width,
        drop_unlabeled=True
    )

    print("\n=== Alignment diagnostics (matching by (patient_id, case_id) as strings) ===")
    print(align_diag)

    print("\n=== Post_y diagnostics ===")
    print(post_diag)

    print("\n=== Pre_v diagnostics ===")
    print(pre_diag)

    print("\n=== Intra_v diagnostics ===")
    print(intra_v_diag)

    print("\n=== Intra_xa diagnostics ===")
    print(intra_diag)

    print("\n=== Static v diagnostics ===")
    print(static_diag)
    print(f"ASA encoding: {asa_encoding} | ethnicity categories (train-fitted): {len(static_enc.ethnicity_categories)}")
    print(f"dim_v = {static_diag['static_dim_v']}")

    # 5) Histogram of sequence lengths
    if plot_length_hist:
        lengths_arr = np.array([s.length for s in sequences], dtype=int)
        plt.figure()
        plt.hist(lengths_arr, bins=30)
        plt.title("Histogram of sequence lengths")
        plt.xlabel("Sequence length (time steps)")
        plt.ylabel("Count")
        plt.show()

    # 6) Build dataset + split (now on final sequences list)
    dataset = SurgeryDataset(sequences)
    n_total = len(dataset)
    n_train = int(round(train_frac * n_total))
    n_train = max(1, min(n_train, n_total - 1))
    n_test = n_total - n_train

    g = torch.Generator().manual_seed(int(seed))
    train_subset, test_subset = random_split(dataset, [n_train, n_test], generator=g)

    # 7) Normalize x (train-split)
    if normalize_x:
        binary_x_cols = set(intra_diag["binary_cols_detected"])
        cumulative_cols = set(intra_diag["cumulative_like_cols"])

        train_seqs = [dataset.sequences[i] for i in train_subset.indices]
        X_train = np.concatenate([s.x for s in train_seqs], axis=0)  # (sumT, dim_x)

        # If you keep cumulative representation, force cumulative-like cols to scale-only
        force_scale = set()
        if not use_incremental_for_cumulative_x:
            force_scale = cumulative_cols

        x_plan = make_normalization_plan_from_matrix(
            X_train=X_train,
            col_names=X_COLS,
            binary_cols=binary_x_cols,
            force_scale_cols=force_scale,
            scale_nonneg_only=scale_nonneg_only
        )
        apply_normalization_inplace_sequences_x(dataset.sequences, x_plan)

        print("\n=== X normalization lists (based on TRAIN split stats) ===")
        print("x <- x / std (scale-only):")
        print(x_plan.scale_cols)
        print("\nx <- (x - mean) / std (z-score):")
        print(x_plan.zscore_cols)
        print("\n(no normalization):")
        print(x_plan.none_cols)

        print("\n=== Negative-value x columns ===")
        print("Negative values in RAW x (before cumulative->incremental):")
        print(intra_diag["negative_x_cols_raw"])
        print("\nNegative values AFTER your chosen representation:")
        print(intra_diag["negative_x_cols_post"])

    # 8) Normalize v (train-split)
    if normalize_v:
        train_seqs = [dataset.sequences[i] for i in train_subset.indices]
        V_train = np.stack([s.v for s in train_seqs], axis=0)  # (n_train, dim_v)

        # Detect binary / onehot columns in v by name:
        # - onehot columns contain "__"
        # - also treat known binary-like source cols as binary by name
        v_cols = static_enc.v_col_names
        v_binary_cols = set([c for c in v_cols if "__" in c])  # onehot -> none

        # Also mark obvious binary flags by source name:
        likely_binary_prefixes = {
            "gender",
            "is_emergency",
            "CABG_oper",
            "CardiacTumor_oper",
            "HeartTransplant_oper",
            "aortic_oper",
            "congenital_oper",
            "valve_oper",
        }
        for c in v_cols:
            base = c.split("__")[0]  # ethnicity__X -> ethnicity
            if base in likely_binary_prefixes:
                v_binary_cols.add(c)

        # Additionally, we can auto-detect binary among non-onehot columns using V_train values:
        # (keep this conservative; onehot already covered)
        for j, c in enumerate(v_cols):
            if c in v_binary_cols:
                continue
            uniq = np.unique(V_train[:, j].astype(np.float32))
            if len(uniq) <= 2 and set(uniq.tolist()).issubset({0.0, 1.0}):
                v_binary_cols.add(c)

        v_plan = make_normalization_plan_from_matrix(
            X_train=V_train,
            col_names=v_cols,
            binary_cols=v_binary_cols,
            force_scale_cols=set(),  # no special forcing beyond binary/onehot
            scale_nonneg_only=scale_nonneg_only
        )
        apply_normalization_inplace_sequences_v(dataset.sequences, v_plan)

        # v negatives diagnostics
        neg_v_cols = []
        for j, c in enumerate(v_cols):
            if np.min(V_train[:, j]) < 0:
                neg_v_cols.append(c)

        print("\n=== V normalization lists (based on TRAIN split stats) ===")
        print("v <- v / std (scale-only):")
        print(v_plan.scale_cols)
        print("\nv <- (v - mean) / std (z-score):")
        print(v_plan.zscore_cols)
        print("\n(no normalization):")
        print(v_plan.none_cols)

        print("\n=== Negative-value v columns (on TRAIN matrix before normalization) ===")
        print(neg_v_cols)

    # 9) Loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size_train,
        shuffle=True,
        collate_fn=lambda b: pad_collate_varlen(b, pad_action=0),
        drop_last=False
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size_test,
        shuffle=False,
        collate_fn=lambda b: pad_collate_varlen(b, pad_action=0),
        drop_last=False
    )

    # 10) Sanity check shapes
    x_pad, a_pad, y, v, lengths, mask = next(iter(train_loader))
    print("\n=== Batch shapes ===")
    print("x_pad:", tuple(x_pad.shape),
          "a_pad:", tuple(a_pad.shape),
          "y:", tuple(y.shape),
          "v:", tuple(v.shape))
    print("lengths:", tuple(lengths.shape), "mask:", tuple(mask.shape))

    return train_loader, test_loader, {
        "align_diag": align_diag,
        "post_diag": post_diag,
        "pre_diag": pre_diag,
        "intra_v_diag": intra_v_diag,
        "intra_diag": intra_diag,
        "static_diag": static_diag,
        "static_enc": {
            "asa_encoding": static_enc.asa_encoding,
            "ethnicity_categories": static_enc.ethnicity_categories,
            "v_col_names": static_enc.v_col_names,
        },
        "n_total": n_total,
        "n_train": n_train,
        "n_test": n_test,
    }


# Example:
if __name__ == "__main__":
    train_loader, test_loader, report = create_loaders(
        use_incremental_for_cumulative_x=True,
        pid_width=None,
        normalize_x=True,
        normalize_v=True,
        scale_nonneg_only=True,
        plot_length_hist=True,
        asa_encoding="onehot",  # or "ordinal"
    )
