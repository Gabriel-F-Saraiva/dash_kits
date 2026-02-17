# app.py
# =============================
# PAINEL DE TORRES (Streamlit)
# - Card do topo = RESULTADO REAL (algoritmo)
# - Simulador de compra + nova coluna "Modelos (SKUs) a comprar" (ceil(faltante / N))
# - Tooltips nas colunas (hover) via column_config
# - Admin: pode subir base (na sessão); demais usuários não
# - Botão "Gerar kits agora" + cache (não recomputa toda hora)
# - Abas: simulador + relatórios (kits_resumo, kits_itens, estoque_restante, falha_proximo_kit)
# =============================

import re
import io
import os
import random
from collections import defaultdict, deque

import pandas as pd
import numpy as np
import streamlit as st


# =============================
# CONFIG GERAL
# =============================
st.set_page_config(page_title="Painel de Torres", layout="wide")

TARGET_MIN_DEFAULT = 10000
TARGET_MAX_DEFAULT = 10090

RULES = {
    "CJ": (22, 25),
    "CK": (8, 12),
    "CO": (20, 28),
    "ES": (2, 3),
    "PF": (10, 15),
    "SEM": (1, 2),
    "PM": (2, 4),
    "C_FEMININO": (10, 15),
    "C_MASCULINO": (2, 4),
    "BR_TRIO": (8, 12),
    "BR_GRANDE": (8, 10),
    "BR_DEMAIS": (60, None),
}

PREFIX_DIRECT = ["CJ", "CK", "CO", "ES", "PF", "SEM", "PM"]
ADJUST_CATS = {"BR_DEMAIS", "CO"}  # categorias usadas como "ajuste fino"

DISPLAY_NAME = {
    "CJ": "CJ",
    "CK": "CK",
    "CO": "CO",
    "ES": "ES",
    "PF": "PF",
    "SEM": "SEM",
    "PM": "PM",
    "C_FEMININO": "C - FEMININO",
    "C_MASCULINO": "C - MASCULINO",
    "BR_TRIO": "BR - TRIO",
    "BR_GRANDE": "BR - GRANDE",
    "BR_DEMAIS": "BR - OUTROS",
}

# Gerador
DEFAULT_MAX_KITS = 200
ATTEMPTS_PER_KIT = 40
TABU_ITERS = 250
TABU_TENURE = 25
NEIGHBORHOOD_SAMPLES = 250
SEED = 7
random.seed(SEED)
np.random.seed(SEED)

# Base default no repo
DEFAULT_BASE_PATH = "base_ativa.xlsx"


# =============================
# CSS (PowerBI-like + inputs legíveis)
# =============================
st.markdown(
    """
    <style>
    :root{
      --bg:#070a0f;
      --panel:#0f1420;
      --panel2:#0b111b;
      --border:#202a3a;
      --border2:#2a3750;
      --text:#e9eef8;
      --muted:#a7b4c7;
      --accent:#5aa9ff;
      --accent2:#7c5cff;
    }

    .stApp { background: radial-gradient(1200px 600px at 20% 0%, #0d1630 0%, var(--bg) 55%); }
    html, body, [class*="css"]  { color: var(--text); }

    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stToolbar"] {visibility: hidden; height: 0px;}
    [data-testid="stDecoration"] {display: none;}

    .block-container { padding-top: 0.85rem; padding-bottom: 2rem; }

    section[data-testid="stSidebar"]{
      background: linear-gradient(180deg, #0b1220 0%, #070a0f 100%);
      border-right: 1px solid var(--border);
    }

    .topbar{
      background: linear-gradient(90deg, rgba(90,169,255,0.18) 0%, rgba(124,92,255,0.12) 45%, rgba(0,0,0,0) 100%);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 14px 16px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.35);
    }
    .topbar-title{
      font-size: 28px;
      font-weight: 800;
      letter-spacing: 0.8px;
      margin: 0;
      line-height: 1.15;
    }
    .topbar-sub{
      color: var(--muted);
      margin-top: 6px;
      font-size: 13px;
    }

    div[data-testid="stMetric"]{
      background: linear-gradient(180deg, var(--panel) 0%, var(--panel2) 100%);
      border: 1px solid var(--border);
      padding: 14px 14px 10px 14px;
      border-radius: 16px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.35);
    }
    div[data-testid="stMetric"] label { color: var(--muted) !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"]{
      font-size: 34px;
      font-weight: 800;
    }

    .stTabs [data-baseweb="tab-list"]{
      gap: 8px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 8px;
    }
    .stTabs [data-baseweb="tab"]{
      background: rgba(255,255,255,0.03);
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 8px 14px;
      color: var(--muted);
    }
    .stTabs [aria-selected="true"]{
      background: linear-gradient(90deg, rgba(90,169,255,0.18), rgba(124,92,255,0.15));
      border: 1px solid var(--border2);
      color: var(--text);
    }

    .stButton>button, .stDownloadButton>button{
      border-radius: 12px !important;
      border: 1px solid var(--border2) !important;
      background: linear-gradient(180deg, rgba(90,169,255,0.16), rgba(90,169,255,0.06)) !important;
      color: var(--text) !important;
      box-shadow: 0 8px 24px rgba(0,0,0,0.35);
    }
    .stButton>button:hover, .stDownloadButton>button:hover{
      border-color: rgba(90,169,255,0.55) !important;
      transform: translateY(-1px);
    }

    div[data-baseweb="input"] input, textarea{
      background: rgba(255,255,255,0.03) !important;
      border: 1px solid var(--border) !important;
      color: var(--text) !important;
      border-radius: 12px !important;
    }

    /* Sidebar inputs com texto ESCURO (legível) */
    section[data-testid="stSidebar"] div[data-baseweb="input"] input{
      color: #0b0f18 !important;
      background: rgba(255,255,255,0.90) !important;
      border: 1px solid rgba(255,255,255,0.18) !important;
    }
    section[data-testid="stSidebar"] div[data-baseweb="input"] input::placeholder{
      color: rgba(11,15,24,0.55) !important;
    }
    section[data-testid="stSidebar"] label{
      color: rgba(233,238,248,0.92) !important;
    }
    section[data-testid="stSidebar"] button[aria-label="Increment"],
    section[data-testid="stSidebar"] button[aria-label="Decrement"]{
      background: rgba(255,255,255,0.85) !important;
      border: 1px solid rgba(255,255,255,0.18) !important;
    }
    section[data-testid="stSidebar"] button[aria-label="Increment"] svg,
    section[data-testid="stSidebar"] button[aria-label="Decrement"] svg{
      fill: #0b0f18 !important;
    }

    .dataframe-shell{
      background: linear-gradient(180deg, var(--panel) 0%, var(--panel2) 100%);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 12px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.35);
    }

    [data-testid="stDataFrame"]{
      border-radius: 12px;
      overflow: hidden;
      border: 1px solid rgba(255,255,255,0.06);
    }
    [data-testid="stDataFrame"] tbody tr:hover{
      background: rgba(90,169,255,0.08) !important;
    }

    hr { border: 0; height: 1px; background: var(--border); margin: 14px 0; }
    .muted { color: var(--muted); font-size: 13px; }
    </style>
    """,
    unsafe_allow_html=True
)


# =============================
# AUTH (admin)
# =============================
def is_admin_logged() -> bool:
    return bool(st.session_state.get("is_admin", False))

def admin_login_box():
    admin_user = st.secrets.get("ADMIN_USER", "admin")
    admin_pass = st.secrets.get("ADMIN_PASSWORD", "admin")

    with st.sidebar:
        st.header("Admin")
        u = st.text_input("Usuário", value="", type="default")
        p = st.text_input("Senha", value="", type="password")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Entrar"):
                if u == admin_user and p == admin_pass:
                    st.session_state["is_admin"] = True
                    st.success("Admin logado.")
                else:
                    st.session_state["is_admin"] = False
                    st.error("Usuário/senha inválidos.")
        with col2:
            if st.button("Sair"):
                st.session_state["is_admin"] = False
                st.info("Saiu.")


# =============================
# BASE PREP
# =============================
def norm_sku(s: str) -> str:
    return re.sub(r"\s+", "", str(s).strip()).upper()

def assign_category(row) -> str:
    sku = row["Sku_norm"]

    if int(row.get("BASE_Corrente_Feminina", 0) or 0) == 1:
        return "C_FEMININO"
    if int(row.get("BASE_Corrente_Masculina", 0) or 0) == 1:
        return "C_MASCULINO"

    if sku.startswith("BR"):
        if int(row.get("BASE_Trio", 0) or 0) == 1:
            return "BR_TRIO"
        if int(row.get("TIPO_Brinco_Grande", 0) or 0) == 1:
            return "BR_GRANDE"
        return "BR_DEMAIS"

    for p in PREFIX_DIRECT:
        if sku.startswith(p):
            return p

    return "OUTROS"

def preparar_base_from_df(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["Sku", "Estoque", "Preco"]:
        if col not in df.columns:
            raise ValueError(f"A planilha precisa ter a coluna '{col}'.")

    df = df.copy()
    df["Sku"] = df["Sku"].astype(str)
    df["Sku_norm"] = df["Sku"].apply(norm_sku)

    df["Estoque"] = pd.to_numeric(df["Estoque"], errors="coerce").fillna(0).astype(int)
    df["Preco"] = pd.to_numeric(df["Preco"], errors="coerce")

    df["categoria"] = df.apply(assign_category, axis=1)

    allowed = set(RULES.keys())
    df = df[(df["Estoque"] > 0) & (df["Preco"].notna()) & (df["categoria"].isin(allowed))].copy()
    return df

@st.cache_data(show_spinner=False)
def load_base_from_bytes(xlsx_bytes: bytes) -> pd.DataFrame:
    df = pd.read_excel(io.BytesIO(xlsx_bytes))
    return preparar_base_from_df(df)

@st.cache_data(show_spinner=False)
def load_default_base_from_repo() -> tuple[pd.DataFrame, str, bytes]:
    if not os.path.exists(DEFAULT_BASE_PATH):
        raise FileNotFoundError(
            f"Não encontrei '{DEFAULT_BASE_PATH}' no repositório. "
            f"Coloque esse arquivo na raiz do repo."
        )
    with open(DEFAULT_BASE_PATH, "rb") as f:
        b = f.read()
    base = load_base_from_bytes(b)
    return base, DEFAULT_BASE_PATH, b

def get_active_base() -> tuple[pd.DataFrame, str, bytes]:
    if "base_bytes" in st.session_state and st.session_state["base_bytes"]:
        b = st.session_state["base_bytes"]
        base = load_base_from_bytes(b)
        name = st.session_state.get("base_name", "upload_admin.xlsx")
        return base, name, b
    return load_default_base_from_repo()


# =============================
# CAPACIDADE TEÓRICA (diagnóstico)
# =============================
def by_sku_table(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["categoria", "Sku_norm"], as_index=False).agg(
        Estoque=("Estoque", "max"),
        Preco=("Preco", "min"),
        Sku=("Sku", "first"),
    )

def max_kits_category_from_stocks(stocks: np.ndarray, m: int) -> int:
    stocks = np.array(stocks, dtype=int)
    stocks = stocks[stocks > 0]
    if len(stocks) < m:
        return 0
    hi = int(stocks.sum() // m)
    lo = 0

    def feasible(k: int) -> bool:
        if k <= 0:
            return True
        return int(np.minimum(stocks, k).sum()) >= k * m

    while lo < hi:
        mid = (lo + hi + 1) // 2
        if feasible(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo

def capacity_table_correct(df: pd.DataFrame) -> pd.DataFrame:
    bs = by_sku_table(df)
    rows = []
    for cat, (mn, mx) in RULES.items():
        sub = bs[bs["categoria"] == cat]
        stocks = sub["Estoque"].to_numpy(dtype=int)
        kits_cat = max_kits_category_from_stocks(stocks, mn)
        rows.append({
            "categoria": cat,
            "Grupo": DISPLAY_NAME.get(cat, cat),
            "min_por_kit": mn,
            "skus_unicos": int(sub["Sku_norm"].nunique()),
            "estoque_total": int(sub["Estoque"].sum()),
            "kits_max_cat": int(kits_cat),
        })
    out = pd.DataFrame(rows)
    out["gargalo"] = out["kits_max_cat"] == out["kits_max_cat"].min()
    return out.sort_values(["kits_max_cat", "Grupo"], ascending=[True, True])

def kits_possible_overall_correct(df: pd.DataFrame) -> tuple[int, str, pd.DataFrame]:
    t = capacity_table_correct(df)
    kits_max = int(t["kits_max_cat"].min()) if len(t) else 0
    gargalos = t.loc[t["kits_max_cat"] == kits_max, "Grupo"].tolist()
    gargalo_str = ", ".join(gargalos) if gargalos else "-"
    return kits_max, gargalo_str, t


# =============================
# SIMULADOR DE COMPRA (com modelos por fórmula correta)
# =============================
def summarize_category(df: pd.DataFrame):
    bs = by_sku_table(df)
    cat = bs.groupby("categoria", as_index=False).agg(
        skus_unicos=("Sku_norm", "nunique"),
        estoque_total=("Estoque", "sum"),
        preco_min=("Preco", "min"),
        preco_med=("Preco", "median"),
        preco_max=("Preco", "max"),
    )

    quantiles = [
        (0.05, "p05"), (0.10, "p10"), (0.25, "p25"), (0.35, "p35"),
        (0.50, "p50"), (0.60, "p60"), (0.75, "p75"), (0.85, "p85"),
        (0.90, "p90"), (0.95, "p95"),
    ]
    for q, name in quantiles:
        cat[f"preco_{name}"] = cat["categoria"].map(
            lambda c: float(np.quantile(bs.loc[bs["categoria"] == c, "Preco"], q))
            if (bs["categoria"] == c).any()
            else np.nan
        )

    cat["min_por_kit"] = cat["categoria"].map(lambda c: RULES[c][0])
    return cat, bs

def min_cost_theoretical(bs: pd.DataFrame) -> float:
    total = 0.0
    for cat, (mn, _) in RULES.items():
        prices = bs.loc[bs["categoria"] == cat, "Preco"].sort_values().to_numpy()
        if len(prices) < mn:
            return np.inf
        total += float(prices[:mn].sum())
    return float(total)

def choose_price_band(direction: str, weight: float, is_adjust: bool):
    if direction == "cheaper":
        if is_adjust:
            return ("p05", "p60", "ajuste-fino barato (P05–P60)")
        if weight >= 0.18:
            return ("p05", "p35", "peso alto: comprar bem barato (P05–P35)")
        if weight >= 0.10:
            return ("p10", "p50", "comprar barato (P10–P50)")
        return ("p10", "p60", "barato-médio (P10–P60)")

    if direction == "pricier":
        if is_adjust:
            return ("p60", "p95", "ajuste-fino mais caro (P60–P95)")
        if weight >= 0.18:
            return ("p50", "p85", "subir valor com controle (P50–P85)")
        return ("p60", "p90", "subir valor (P60–P90)")

    if is_adjust:
        return ("p10", "p90", "ajuste amplo (P10–P90)")
    return ("p25", "p75", "faixa padrão (P25–P75)")

def shortage_slots_for_target(stocks: np.ndarray, m: int, k: int) -> int:
    stocks = np.array(stocks, dtype=int)
    stocks = stocks[stocks > 0]
    have = int(np.minimum(stocks, k).sum())
    need = int(k * m)
    return max(0, need - have)

def simulator_purchase_table(df: pd.DataFrame, target_kits: int, target_min: float, target_max: float):
    cat, bs = summarize_category(df)

    min_cost = min_cost_theoretical(bs)
    if np.isinf(min_cost):
        direction = "neutral"
    elif min_cost > target_max:
        direction = "cheaper"
    elif min_cost < target_min:
        direction = "pricier"
    else:
        direction = "neutral"

    cat["contrib"] = cat["preco_med"] * cat["min_por_kit"]
    contrib_sum = float(cat["contrib"].sum()) if float(cat["contrib"].sum()) > 0 else 1.0
    cat["peso"] = cat["contrib"] / contrib_sum

    faltas = []
    lo_list, hi_list, label_list = [], [], []
    for _, r in cat.iterrows():
        c = r["categoria"]
        mn = int(r["min_por_kit"])
        stocks = bs.loc[bs["categoria"] == c, "Estoque"].to_numpy(dtype=int)
        falta_slots = shortage_slots_for_target(stocks, mn, int(target_kits))
        faltas.append(falta_slots)

        is_adjust = c in ADJUST_CATS
        lo, hi, label = choose_price_band(direction, float(r["peso"]), is_adjust)
        lo_list.append(lo); hi_list.append(hi); label_list.append(label)

    cat["falta_slots"] = pd.Series(faltas, index=cat.index).astype(int)
    cat["estrategia_preco"] = label_list
    cat["preco_sugerido_de"] = [round(float(r[f"preco_{lo}"]), 2) for r, lo in zip(cat.to_dict("records"), lo_list)]
    cat["preco_sugerido_ate"] = [round(float(r[f"preco_{hi}"]), 2) for r, hi in zip(cat.to_dict("records"), hi_list)]
    cat["preco_sugerido_medio"] = (cat["preco_sugerido_de"] + cat["preco_sugerido_ate"]) / 2.0

    cat["requerido_slots"] = int(target_kits) * cat["min_por_kit"]
    cat["indice_faltante"] = np.where(cat["requerido_slots"] > 0, cat["falta_slots"] / cat["requerido_slots"], 0.0)
    cat["custo_reposicao"] = cat["falta_slots"] * cat["preco_sugerido_medio"]

    # ==========================
    # NOVO (CORRETO): modelos por limitação 1x por kit
    # Cada SKU pode contribuir no máximo N unidades úteis (1 por kit).
    # Portanto, modelos mínimos = ceil(faltante_slots / N)
    # E unidades sugeridas por SKU = ceil(faltante / modelos), limitado a N (não agrega acima de N).
    # ==========================
    N = max(int(target_kits), 1)
    cat["modelos_sku_a_comprar"] = np.ceil(cat["falta_slots"] / N).astype(int)

    cat["unid_sugeridas_por_sku"] = np.where(
        cat["modelos_sku_a_comprar"] > 0,
        np.ceil(cat["falta_slots"] / cat["modelos_sku_a_comprar"]),
        0
    ).astype(int)
    cat["unid_sugeridas_por_sku"] = np.minimum(cat["unid_sugeridas_por_sku"], N).astype(int)

    out = pd.DataFrame({
        "Grupo": cat["categoria"].map(lambda x: DISPLAY_NAME.get(x, x)),
        "Estoque": cat["estoque_total"].astype(int),
        "Faltante para a meta": cat["falta_slots"].astype(int),
        "Modelos (SKUs) a comprar": cat["modelos_sku_a_comprar"].astype(int),
        "Unid. sugeridas por SKU": cat["unid_sugeridas_por_sku"].astype(int),
        "Índice faltante por grupo": cat["indice_faltante"].astype(float),
        "Custo de reposição": cat["custo_reposicao"].astype(float),
        "Preço sugerido (de)": cat["preco_sugerido_de"].astype(float),
        "Preço sugerido (até)": cat["preco_sugerido_ate"].astype(float),
        "Estratégia de preço": cat["estrategia_preco"].astype(str),
    }).sort_values(["Faltante para a meta", "Grupo"], ascending=[False, True])

    total_row = pd.DataFrame([{
        "Grupo": "Total",
        "Estoque": int(out["Estoque"].sum()),
        "Faltante para a meta": int(out["Faltante para a meta"].sum()),
        "Modelos (SKUs) a comprar": int(out["Modelos (SKUs) a comprar"].sum()),
        "Unid. sugeridas por SKU": np.nan,
        "Índice faltante por grupo": float(out["Índice faltante por grupo"].mean()) if len(out) else 0.0,
        "Custo de reposição": float(out["Custo de reposição"].sum()),
        "Preço sugerido (de)": np.nan,
        "Preço sugerido (até)": np.nan,
        "Estratégia de preço": "",
    }])
    out = pd.concat([out, total_row], ignore_index=True)

    return out, direction, float(min_cost)


# =============================
# EXPORT HELPERS
# =============================
def df_to_excel_bytes(sheets: dict) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, index=False, sheet_name=name[:31])
    bio.seek(0)
    return bio.getvalue()

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def fmt_brl(x: float) -> str:
    return f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


# =============================
# KIT GENERATOR (VALUE-FIRST + TABU + FALHA)
# =============================
def build_structures(base: pd.DataFrame):
    stock0 = base.groupby("Sku_norm")["Estoque"].max().to_dict()
    price = base.groupby("Sku_norm")["Preco"].min().to_dict()
    cat_of = base.groupby("Sku_norm")["categoria"].first().to_dict()

    pools = {}
    for cat in RULES.keys():
        d = base[base["categoria"] == cat].drop_duplicates("Sku_norm").copy()
        d.sort_values(["Preco", "Estoque"], ascending=[True, True], inplace=True)
        pools[cat] = d["Sku_norm"].tolist()

    return stock0, price, cat_of, pools

def objective(total: float, tmin: float, tmax: float) -> float:
    mid = (tmin + tmax) / 2
    if tmin <= total <= tmax:
        return abs(total - mid)
    if total < tmin:
        return 10_000 + (tmin - total)
    return 10_000 + (total - tmax)

def can_add_cat(counts, cat):
    mn, mx = RULES[cat]
    return True if mx is None else counts[cat] < mx

def pick_k_skus(cat, k, used, stock, pools):
    cands = [s for s in pools[cat] if stock.get(s, 0) > 0 and s not in used]
    if len(cands) < k:
        return None
    return cands[:k]

def pick_best_fit(cat, used, stock, pools, price, current_total, tmin, tmax):
    max_add = tmax - current_total
    if max_add <= 0:
        return None

    gap = tmin - current_total
    cands = [
        s for s in pools[cat]
        if stock.get(s, 0) > 0 and s not in used and float(price[s]) <= max_add
    ]
    if not cands:
        return None

    if gap > 0:
        return min(cands, key=lambda s: abs(float(price[s]) - gap))
    return min(cands, key=lambda s: objective(current_total + float(price[s]), tmin, tmax))

def greedy_build(stock, pools, price, cat_of, tmin, tmax):
    used = set()
    counts = defaultdict(int)
    selected = []
    total = 0.0

    for cat, (mn, mx) in RULES.items():
        pick = pick_k_skus(cat, mn, used, stock, pools)
        if pick is None:
            return None
        for s in pick:
            used.add(s); selected.append(s); counts[cat] += 1; total += float(price[s])

    if total > tmax:
        return None

    cats_order = ["BR_DEMAIS", "CO"] + [c for c in RULES.keys() if c not in ("BR_DEMAIS", "CO")]
    step = 0
    while total < tmin and step < 1200:
        step += 1
        chosen = None
        chosen_cat = None

        for cat in cats_order:
            if not can_add_cat(counts, cat):
                continue
            s = pick_best_fit(cat, used, stock, pools, price, total, tmin, tmax)
            if s is not None:
                chosen = s
                chosen_cat = cat
                break

        if chosen is None:
            return None

        used.add(chosen); selected.append(chosen); counts[chosen_cat] += 1; total += float(price[chosen])
        if total > tmax:
            return None

    return {"skus": selected, "used": used, "counts": dict(counts), "total": total}

def tabu_improve(sol, stock, pools, price, cat_of, tmin, tmax,
                max_iters=TABU_ITERS, tenure=TABU_TENURE, samples=NEIGHBORHOOD_SAMPLES):
    skus = sol["skus"][:]
    used = set(skus)
    counts = defaultdict(int, sol["counts"])
    total = float(sol["total"])
    tabu = deque(maxlen=tenure)

    def is_tabu(m): return m in tabu
    def add_tabu(m): tabu.append(m)

    def kit_items(cat):
        return [s for s in skus if cat_of.get(s) == cat]

    def best_in_cat_for_swap(cat, out_sku, current_total):
        max_add = tmax - (current_total - float(price[out_sku]))
        cands = [
            s for s in pools[cat]
            if stock.get(s, 0) > 0 and s not in used and float(price[s]) <= max_add
        ]
        if not cands:
            return None
        return min(cands[:300], key=lambda s: objective(current_total - float(price[out_sku]) + float(price[s]), tmin, tmax))

    best_total = total
    best_skus = skus[:]
    best_counts = counts.copy()
    best_obj = objective(total, tmin, tmax)

    for _ in range(max_iters):
        best_neighbor = None
        best_neighbor_obj = None
        best_move = None

        for _ in range(samples):
            r = random.random()
            if r < 0.65:
                move_type, cat = "swap", "BR_DEMAIS"
            elif r < 0.88:
                move_type, cat = "swap", "CO"
            elif r < 0.95:
                move_type, cat = "add", "BR_DEMAIS"
            else:
                move_type, cat = "remove", "BR_DEMAIS"

            if move_type == "swap":
                items = kit_items(cat)
                if not items:
                    continue
                out_sku = random.choice(items)
                in_sku = best_in_cat_for_swap(cat, out_sku, total)
                if in_sku is None:
                    continue
                move = (move_type, cat, out_sku, in_sku)
                if is_tabu(move):
                    continue
                new_total = total - float(price[out_sku]) + float(price[in_sku])
                new_obj = objective(new_total, tmin, tmax)
                if best_neighbor_obj is None or new_obj < best_neighbor_obj:
                    best_neighbor_obj = new_obj
                    best_neighbor = ("swap", out_sku, in_sku, new_total)
                    best_move = move

            elif move_type == "add":
                if not can_add_cat(counts, "BR_DEMAIS"):
                    continue
                s = pick_best_fit("BR_DEMAIS", used, stock, pools, price, total, tmin, tmax)
                if s is None:
                    continue
                move = (move_type, "BR_DEMAIS", None, s)
                if is_tabu(move):
                    continue
                new_total = total + float(price[s])
                if new_total > tmax:
                    continue
                new_obj = objective(new_total, tmin, tmax)
                if best_neighbor_obj is None or new_obj < best_neighbor_obj:
                    best_neighbor_obj = new_obj
                    best_neighbor = ("add", None, s, new_total)
                    best_move = move

            else:
                min_br = RULES["BR_DEMAIS"][0]
                if counts["BR_DEMAIS"] <= min_br:
                    continue
                items = kit_items("BR_DEMAIS")
                if not items:
                    continue
                out_sku = random.choice(items)
                move = (move_type, "BR_DEMAIS", out_sku, None)
                if is_tabu(move):
                    continue
                new_total = total - float(price[out_sku])
                new_obj = objective(new_total, tmin, tmax)
                if best_neighbor_obj is None or new_obj < best_neighbor_obj:
                    best_neighbor_obj = new_obj
                    best_neighbor = ("remove", out_sku, None, new_total)
                    best_move = move

        if best_neighbor is None:
            break

        kind, out_sku, in_sku, new_total = best_neighbor
        if kind == "swap":
            skus.remove(out_sku); used.remove(out_sku)
            skus.append(in_sku); used.add(in_sku)
            total = new_total
        elif kind == "add":
            skus.append(in_sku); used.add(in_sku)
            counts["BR_DEMAIS"] += 1
            total = new_total
        else:
            skus.remove(out_sku); used.remove(out_sku)
            counts["BR_DEMAIS"] -= 1
            total = new_total

        add_tabu(best_move)

        cur_obj = objective(total, tmin, tmax)
        if cur_obj < best_obj:
            best_obj = cur_obj
            best_total = total
            best_skus = skus[:]
            best_counts = counts.copy()

        if tmin <= total <= tmax and best_obj <= 1.0:
            break

    return {"skus": best_skus, "used": set(best_skus), "counts": dict(best_counts), "total": float(best_total)}

def try_build_one_with_reason(stock, pools, price, tmin, tmax):
    used = set()
    counts = defaultdict(int)
    total = 0.0

    for cat, (mn, mx) in RULES.items():
        pick = pick_k_skus(cat, mn, used, stock, pools)
        if pick is None:
            cands = [s for s in pools[cat] if stock.get(s, 0) > 0 and s not in used]
            return None, f"Falhou nos mínimos: {cat} precisa {mn}, disponíveis {len(cands)}"
        for s in pick:
            used.add(s); counts[cat] += 1; total += float(price[s])

    if total > tmax:
        return None, f"Mínimos estouram teto: total_minimos={total:.2f} > {tmax}"

    cats_order = ["BR_DEMAIS", "CO"] + [c for c in RULES.keys() if c not in ("BR_DEMAIS", "CO")]
    step = 0
    while total < tmin and step < 1200:
        step += 1
        chosen = None
        for cat in cats_order:
            if not can_add_cat(counts, cat):
                continue
            s = pick_best_fit(cat, used, stock, pools, price, total, tmin, tmax)
            if s is not None:
                chosen = s
                break
        if chosen is None:
            return None, f"Não conseguiu completar: total={total:.2f}, falta={tmin-total:.2f}, nenhum item cabe até {tmax}"
        used.add(chosen); total += float(price[chosen])
        if total > tmax:
            return None, f"Estourou teto ao completar: total={total:.2f} > {tmax}"

    if not (tmin <= total <= tmax):
        return None, f"Terminou fora da faixa: total={total:.2f}"

    return {"total": total, "counts": dict(counts), "skus": list(used)}, "OK"

def diagnose_next_kit(stock, pools, price):
    rows = []

    for cat, (mn, mx) in RULES.items():
        cands = [s for s in pools[cat] if stock.get(s, 0) > 0]
        skus_disp = len(cands)
        estoque_total = sum(stock.get(s, 0) for s in cands)
        status = "OK" if skus_disp >= mn else "FALTA_SKU"
        rows.append({
            "tipo": "minimos_viabilidade",
            "categoria": cat,
            "min": mn,
            "max": (mx if mx is not None else "∞"),
            "skus_disponiveis": skus_disp,
            "estoque_total_categoria": estoque_total,
            "status": status
        })

    used = set()
    total_min = 0.0
    min_break = None
    min_details = []

    for cat, (mn, mx) in RULES.items():
        cands = [s for s in pools[cat] if stock.get(s, 0) > 0 and s not in used]
        if len(cands) < mn:
            min_break = f"Quebrou em {cat}: precisa {mn}, tem {len(cands)} (sem repetir SKU)."
            break
        pick = cands[:mn]
        cat_sum = sum(float(price[s]) for s in pick)
        total_min += cat_sum
        used.update(pick)
        min_details.append((cat, mn, cat_sum, total_min))

    rows.append({
        "tipo": "custo_minimo_mins",
        "categoria": "-",
        "min": "-",
        "max": "-",
        "skus_disponiveis": "-",
        "estoque_total_categoria": "-",
        "status": f"TOTAL_MIN={total_min:.2f} | " + ("OK" if (min_break is None) else "IMPOSSIVEL") + (f" | {min_break}" if min_break else "")
    })

    for cat, mn, cat_sum, acc in min_details:
        rows.append({
            "tipo": "custo_minimo_detalhe",
            "categoria": cat,
            "min": mn,
            "max": "-",
            "skus_disponiveis": "-",
            "estoque_total_categoria": "-",
            "status": f"soma_cat={cat_sum:.2f} | acumulado={acc:.2f}"
        })

    return pd.DataFrame(rows)


# =============================
# CACHE DO GERADOR (relatórios)
# =============================
@st.cache_data(show_spinner=False)
def generate_kits_reports(base_bytes: bytes, tmin: float, tmax: float, max_kits: int) -> dict:
    base = load_base_from_bytes(base_bytes)
    stock0, price, cat_of, pools = build_structures(base)

    stock = dict(stock0)
    kits = []
    failure_info = None

    for kit_id in range(1, max_kits + 1):
        best = None
        for _ in range(ATTEMPTS_PER_KIT):
            sol = greedy_build(stock, pools, price, cat_of, tmin, tmax)
            if sol is None:
                continue
            sol2 = tabu_improve(sol, stock, pools, price, cat_of, tmin, tmax)
            if tmin <= sol2["total"] <= tmax:
                best = sol2
                break

        if best is None:
            _, reason = try_build_one_with_reason(stock, pools, price, tmin, tmax)
            failure_info = {"kit_que_falhou": kit_id, "motivo": reason}
            break

        for s in best["skus"]:
            stock[s] -= 1

        best["kit_id"] = kit_id
        kits.append(best)

    base_lookup = base.drop_duplicates("Sku_norm").set_index("Sku_norm")

    rows_items = []
    for k in kits:
        for s in k["skus"]:
            r = base_lookup.loc[s]
            rows_items.append({
                "kit_id": k["kit_id"],
                "Sku": r["Sku"],
                "Sku_norm": s,
                "categoria": r["categoria"],
                "Preco": float(r["Preco"]),
            })
    kits_itens = pd.DataFrame(rows_items)

    summary_rows = []
    for k in kits:
        row = {"kit_id": k["kit_id"], "total_preco": float(k["total"]), "qtd_itens": int(len(k["skus"]))}
        for cat in RULES:
            row[f"qtd_{cat}"] = int(k["counts"].get(cat, 0))
        summary_rows.append(row)
    kits_resumo = pd.DataFrame(summary_rows)

    estoque_restante = (
        pd.DataFrame([{"Sku_norm": s, "Estoque_restante": q} for s, q in stock.items()])
        .merge(base[["Sku_norm", "Sku", "categoria", "Preco"]].drop_duplicates("Sku_norm"),
               on="Sku_norm", how="left")
        .sort_values(["categoria", "Sku_norm"])
    )

    falha_df = diagnose_next_kit(stock, pools, price)
    if failure_info:
        header = pd.DataFrame([{
            "tipo": "resumo_falha",
            "categoria": "-",
            "min": "-",
            "max": "-",
            "skus_disponiveis": "-",
            "estoque_total_categoria": "-",
            "status": f"Kit {failure_info['kit_que_falhou']} falhou | {failure_info['motivo']}"
        }])
        falha_df = pd.concat([header, falha_df], ignore_index=True)

    return {
        "kits_resumo": kits_resumo,
        "kits_itens": kits_itens,
        "estoque_restante": estoque_restante,
        "falha_proximo_kit": falha_df,
        "qtd_kits": len(kits),
        "failure_info": failure_info,
    }

@st.cache_data(show_spinner=False)
def compute_real_kits_count(base_bytes: bytes, tmin: float, tmax: float, max_kits: int) -> int:
    reports = generate_kits_reports(base_bytes, tmin, tmax, max_kits)
    return int(reports.get("qtd_kits", 0))


# =============================
# UI - Sidebar
# =============================
admin_login_box()

with st.sidebar:
    st.header("Configurações")
    target_min = st.number_input("Preço mínimo do kit", value=TARGET_MIN_DEFAULT, step=10)
    target_max = st.number_input("Preço máximo do kit", value=TARGET_MAX_DEFAULT, step=10)

    st.divider()
    st.header("Simulador de compra")
    desired_kits = st.slider("Quantidade de torres", min_value=1, max_value=500, value=100, step=1)

    st.divider()
    st.header("Geração de kits")
    max_kits = st.number_input("Gerar até (máx kits)", min_value=1, max_value=500, value=DEFAULT_MAX_KITS, step=10)

    if is_admin_logged():
        st.divider()
        st.header("Atualizar base (admin)")
        up = st.file_uploader("Enviar nova base (xlsx)", type=["xlsx"], key="admin_upload")
        if up is not None:
            st.session_state["base_bytes"] = up.getvalue()
            st.session_state["base_name"] = up.name
            st.success("Base carregada para esta sessão.")
            st.info("Para persistir permanentemente, substitua o arquivo 'base_ativa.xlsx' no repositório.")


# =============================
# MAIN
# =============================
try:
    base_df, base_name, base_bytes = get_active_base()
except Exception as e:
    st.error(str(e))
    st.stop()

kits_teorico, gargalo, _ = kits_possible_overall_correct(base_df)
kits_real = compute_real_kits_count(base_bytes, float(target_min), float(target_max), int(max_kits))

c1, c2 = st.columns([2.6, 1.0])
with c1:
    st.markdown(
        f"""
        <div class="topbar">
          <div class="topbar-title">PAINEL DE TORRES</div>
          <div class="topbar-sub">
            Base ativa: <b>{base_name}</b>
            &nbsp;|&nbsp; Faixa: <b>{fmt_brl(float(target_min))}</b> a <b>{fmt_brl(float(target_max))}</b>
            &nbsp;|&nbsp; Gargalo(s): <b>{gargalo}</b>
            &nbsp;|&nbsp; Teórico (estoque): <b>{kits_teorico}</b>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )
with c2:
    st.metric("Quantidade de torres atualmente", kits_real)

st.markdown("<hr/>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Simulador de compra",
    "Kits resumo",
    "Kits itens",
    "Estoque restante",
    "Falha próximo kit"
])


# =============================
# TAB 1 - Simulador
# =============================
with tab1:
    left, right = st.columns([1.15, 2.85])

    sim_table, _, _ = simulator_purchase_table(base_df, int(desired_kits), float(target_min), float(target_max))

    with left:
        st.subheader("Ações")
        st.markdown('<div class="muted">Altere a quantidade de torres na sidebar para recalcular a tabela.</div>', unsafe_allow_html=True)

        st.download_button(
            "Baixar Simulador (Excel)",
            data=df_to_excel_bytes({"simulador_compra": sim_table}),
            file_name="simulador_compra.xlsx",
        )
        st.download_button(
            "Baixar Simulador (CSV)",
            data=df_to_csv_bytes(sim_table),
            file_name="simulador_compra.csv",
        )

        st.divider()
        st.subheader("Gerar kits")
        st.markdown('<div class="muted">Gera os relatórios abaixo usando o algoritmo (value-first + tabu).</div>', unsafe_allow_html=True)

        if st.button("Gerar kits agora"):
            st.session_state["last_gen"] = generate_kits_reports(
                base_bytes, float(target_min), float(target_max), int(max_kits)
            )
            st.success(f"Kits gerados: {st.session_state['last_gen']['qtd_kits']}")

        if "last_gen" in st.session_state:
            st.info(f"Kits gerados: {st.session_state['last_gen']['qtd_kits']}")

        if st.button("Limpar cache dos kits"):
            if "last_gen" in st.session_state:
                del st.session_state["last_gen"]
            generate_kits_reports.clear()
            compute_real_kits_count.clear()
            st.info("Cache limpo (kits + métrica do topo).")

    with right:
        st.subheader("Simulador de compra")

        display_df = sim_table.copy()
        display_df["Índice faltante por grupo"] = display_df["Índice faltante por grupo"].apply(
            lambda x: f"{x*100:.2f}%" if pd.notna(x) else ""
        )
        display_df["Custo de reposição"] = display_df["Custo de reposição"].apply(
            lambda x: fmt_brl(x) if pd.notna(x) else ""
        )
        display_df["Preço sugerido (de)"] = display_df["Preço sugerido (de)"].apply(
            lambda x: fmt_brl(x) if pd.notna(x) else ""
        )
        display_df["Preço sugerido (até)"] = display_df["Preço sugerido (até)"].apply(
            lambda x: fmt_brl(x) if pd.notna(x) else ""
        )

        st.markdown('<div class="dataframe-shell">', unsafe_allow_html=True)
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Grupo": st.column_config.TextColumn(
                    "Grupo",
                    help="Categoria do kit (ex.: CJ, CO, BR - OUTROS...)."
                ),
                "Estoque": st.column_config.NumberColumn(
                    "Estoque",
                    help="Estoque total disponível no grupo (somando todas as unidades do grupo)."
                ),
                "Faltante para a meta": st.column_config.NumberColumn(
                    "Faltante para a meta",
                    help=(
                        "Quantos 'slots' de itens faltam nesse grupo para montar N kits respeitando o mínimo por kit.\n\n"
                        "Cálculo: faltante = max(0, N*min_por_kit − Σ min(estoque_sku, N))."
                    )
                ),
                "Modelos (SKUs) a comprar": st.column_config.NumberColumn(
                    "Modelos (SKUs) a comprar",
                    help=(
                        "Estimativa mínima de SKUs/modelos diferentes a comprar para cobrir o faltante.\n\n"
                        "Como 1 SKU pode aparecer no máximo 1x por kit, ao longo de N kits cada SKU contribui no máximo N unidades úteis.\n"
                        "Cálculo (correto): ceil(faltante_slots / N)."
                    )
                ),
                "Unid. sugeridas por SKU": st.column_config.NumberColumn(
                    "Unid. sugeridas por SKU",
                    help=(
                        "Sugestão de unidades por SKU novo para cobrir o faltante com os modelos estimados.\n"
                        "Cálculo: ceil(faltante_slots / modelos), limitado a N (acima de N não ajuda a meta)."
                    )
                ),
                "Índice faltante por grupo": st.column_config.TextColumn(
                    "Índice faltante por grupo",
                    help="Percentual do mínimo necessário que está faltando: faltante / (N * min_por_kit)."
                ),
                "Custo de reposição": st.column_config.TextColumn(
                    "Custo de reposição",
                    help=(
                        "Estimativa de custo para cobrir o faltante do grupo.\n\n"
                        "Cálculo: custo = faltante * preço_médio_sugerido, onde preço_médio_sugerido = (de + até)/2."
                    )
                ),
                "Preço sugerido (de)": st.column_config.TextColumn(
                    "Preço sugerido (de)",
                    help="Faixa inferior sugerida para compra (percentil do preço do grupo)."
                ),
                "Preço sugerido (até)": st.column_config.TextColumn(
                    "Preço sugerido (até)",
                    help="Faixa superior sugerida para compra (percentil do preço do grupo)."
                ),
                "Estratégia de preço": st.column_config.TextColumn(
                    "Estratégia de preço",
                    help=(
                        "Explica por que a faixa de preço foi sugerida (baratear/encarecer/ajuste fino),\n"
                        "considerando o impacto do grupo no custo do kit e a faixa alvo (R$ min–max)."
                    )
                ),
            }
        )
        st.markdown("</div>", unsafe_allow_html=True)


# =============================
# RELATÓRIOS (gerados pelo botão)
# =============================
def render_report(tab, key: str, title: str):
    with tab:
        st.subheader(title)

        if "last_gen" not in st.session_state:
            st.info("Clique em **Gerar kits agora** na aba 'Simulador de compra' para montar os relatórios.")
            return

        df = st.session_state["last_gen"].get(key)
        if df is None:
            st.warning("Relatório não disponível.")
            return

        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.download_button("Baixar Excel", data=df_to_excel_bytes({key: df}), file_name=f"{key}.xlsx")
        with col_b:
            st.download_button("Baixar CSV", data=df_to_csv_bytes(df), file_name=f"{key}.csv")

        st.markdown('<div class="dataframe-shell">', unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

render_report(tab2, "kits_resumo", "Kits resumo")
render_report(tab3, "kits_itens", "Kits itens")
render_report(tab4, "estoque_restante", "Estoque restante")
render_report(tab5, "falha_proximo_kit", "Falha próximo kit")
