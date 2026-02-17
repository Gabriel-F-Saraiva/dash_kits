import re
import io
import pandas as pd
import numpy as np
import streamlit as st

# -----------------------------
# CONFIG
# -----------------------------
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
ADJUST_CATS = {"BR_DEMAIS", "CO"}

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

# -----------------------------
# CSS (visual mais "painel")
# -----------------------------
st.markdown(
    """
    <style>
    /* fundo */
    .stApp { background: #088EA8; }
    /* textos */
    html, body, [class*="css"]  { color: #020024; }
    /* remove "padding" gigante do topo */
    .block-container { padding-top: 1.0rem; }
    /* cards / metric */
    div[data-testid="stMetric"] { background: #121212; border: 1px solid #2a2a2a; padding: 12px; border-radius: 10px; }
    /* dataframe */
    .stDataFrame { background: #0b0b0b; }
    /* separadores */
    hr { border: 0; height: 1px; background: #2a2a2a; }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# HELPERS
# -----------------------------
def norm_sku(s: str) -> str:
    return re.sub(r"\s+", "", str(s).strip()).upper()

def assign_category(row) -> str:
    sku = row["Sku_norm"]

    # Correntes por flags
    if int(row.get("BASE_Corrente_Feminina", 0) or 0) == 1:
        return "C_FEMININO"
    if int(row.get("BASE_Corrente_Masculina", 0) or 0) == 1:
        return "C_MASCULINO"

    # BR por flags
    if sku.startswith("BR"):
        if int(row.get("BASE_Trio", 0) or 0) == 1:
            return "BR_TRIO"
        if int(row.get("TIPO_Brinco_Grande", 0) or 0) == 1:
            return "BR_GRANDE"
        return "BR_DEMAIS"

    # Prefixos diretos
    for p in PREFIX_DIRECT:
        if sku.startswith(p):
            return p

    return "OUTROS"

@st.cache_data(show_spinner=False)
def load_base(file) -> pd.DataFrame:
    df = pd.read_excel(file)
    for col in ["Sku", "Estoque", "Preco"]:
        if col not in df.columns:
            raise ValueError(f"A planilha precisa ter a coluna '{col}'.")

    df["Sku"] = df["Sku"].astype(str)
    df["Sku_norm"] = df["Sku"].apply(norm_sku)
    df["Estoque"] = pd.to_numeric(df["Estoque"], errors="coerce").fillna(0).astype(int)
    df["Preco"] = pd.to_numeric(df["Preco"], errors="coerce")
    df["categoria"] = df.apply(assign_category, axis=1)

    allowed = set(RULES.keys())
    df = df[(df["Estoque"] > 0) & (df["Preco"].notna()) & (df["categoria"].isin(allowed))].copy()
    return df

def by_sku_table(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["categoria", "Sku_norm"], as_index=False).agg(
        Estoque=("Estoque", "max"),
        Preco=("Preco", "min"),
        Sku=("Sku", "first"),
    )

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
        (0.05, "p05"),
        (0.10, "p10"),
        (0.25, "p25"),
        (0.35, "p35"),
        (0.50, "p50"),
        (0.60, "p60"),
        (0.75, "p75"),
        (0.85, "p85"),
        (0.90, "p90"),
        (0.95, "p95"),
    ]
    for q, name in quantiles:
        cat[f"preco_{name}"] = cat["categoria"].map(
            lambda c: float(np.quantile(bs.loc[bs["categoria"] == c, "Preco"], q))
            if (bs["categoria"] == c).any()
            else np.nan
        )

    cat["min_por_kit"] = cat["categoria"].map(lambda c: RULES[c][0])
    cat["max_por_kit"] = cat["categoria"].map(lambda c: RULES[c][1] if RULES[c][1] is not None else np.inf)
    return cat, bs

# -----------------------------
# CAPACIDADE CORRETA (repetição entre kits conforme estoque)
# -----------------------------
def max_kits_category_from_stocks(stocks: np.ndarray, m: int) -> int:
    """
    Máximo k tal que sum(min(stocks, k)) >= k*m
    """
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

def shortage_slots_for_target(stocks: np.ndarray, m: int, k: int) -> int:
    stocks = np.array(stocks, dtype=int)
    stocks = stocks[stocks > 0]
    have = int(np.minimum(stocks, k).sum())
    need = int(k * m)
    return max(0, need - have)

def capacity_table_correct(df: pd.DataFrame) -> pd.DataFrame:
    bs = by_sku_table(df)
    rows = []
    for cat, (mn, mx) in RULES.items():
        sub = bs[bs["categoria"] == cat]
        stocks = sub["Estoque"].to_numpy(dtype=int)
        kits_cat = max_kits_category_from_stocks(stocks, mn)

        rows.append({
            "categoria": cat,
            "grupo": DISPLAY_NAME.get(cat, cat),
            "min_por_kit": mn,
            "max_por_kit": (mx if mx is not None else np.inf),
            "skus_unicos": int(sub["Sku_norm"].nunique()),
            "estoque_total": int(sub["Estoque"].sum()),
            "kits_max_cat": int(kits_cat),
            "preco_min": float(sub["Preco"].min()) if len(sub) else np.nan,
            "preco_med": float(sub["Preco"].median()) if len(sub) else np.nan,
            "preco_max": float(sub["Preco"].max()) if len(sub) else np.nan,
        })

    out = pd.DataFrame(rows)
    out["max_por_kit"] = out["max_por_kit"].replace([np.inf], ["∞"])
    out["gargalo"] = out["kits_max_cat"] == out["kits_max_cat"].min()
    return out.sort_values(["kits_max_cat", "grupo"], ascending=[True, True])

def kits_possible_overall_correct(df: pd.DataFrame) -> tuple[int, str, pd.DataFrame]:
    t = capacity_table_correct(df)
    kits_max = int(t["kits_max_cat"].min()) if len(t) else 0
    gargalos = t.loc[t["kits_max_cat"] == kits_max, "grupo"].tolist()
    gargalo_str = ", ".join(gargalos) if gargalos else "-"
    return kits_max, gargalo_str, t

# -----------------------------
# PREÇO INTELIGENTE
# -----------------------------
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

def simulator_purchase_table(df: pd.DataFrame, target_kits: int, target_min: float, target_max: float) -> tuple[pd.DataFrame, str, float]:
    """
    Tabela estilo "Simulador de compra":
    - Grupo
    - Estoque (unidades)
    - Faltante para a meta (slots)
    - Índice faltante por grupo
    - Custo de reposição (estimado)
    + preço sugerido e estratégia
    """
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

    # peso relativo
    cat["contrib"] = cat["preco_med"] * cat["min_por_kit"]
    contrib_sum = float(cat["contrib"].sum()) if float(cat["contrib"].sum()) > 0 else 1.0
    cat["peso"] = cat["contrib"] / contrib_sum

    # falta_slots e preço sugerido
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

    # índice faltante (proporção)
    cat["requerido_slots"] = int(target_kits) * cat["min_por_kit"]
    cat["indice_faltante"] = np.where(
        cat["requerido_slots"] > 0,
        cat["falta_slots"] / cat["requerido_slots"],
        0.0
    )

    # custo reposição (estimado) = falta_slots * preço_sugerido_medio
    cat["custo_reposicao"] = cat["falta_slots"] * cat["preco_sugerido_medio"]

    # montar tabela final
    out = pd.DataFrame({
        "Grupo": cat["categoria"].map(lambda x: DISPLAY_NAME.get(x, x)),
        "Estoque": cat["estoque_total"].astype(int),
        "Faltante para a meta": cat["falta_slots"].astype(int),
        "Índice faltante por grupo": cat["indice_faltante"].astype(float),
        "Custo de reposição": cat["custo_reposicao"].astype(float),
        "Preço sugerido (de)": cat["preco_sugerido_de"].astype(float),
        "Preço sugerido (até)": cat["preco_sugerido_ate"].astype(float),
        "Estratégia de preço": cat["estrategia_preco"].astype(str),
    })

    # Ordena: quem falta mais primeiro
    out = out.sort_values(["Faltante para a meta", "Grupo"], ascending=[False, True])

    # linha total
    total_row = pd.DataFrame([{
        "Grupo": "Total",
        "Estoque": int(out["Estoque"].sum()),
        "Faltante para a meta": int(out["Faltante para a meta"].sum()),
        "Índice faltante por grupo": float(out["Índice faltante por grupo"].mean()) if len(out) else 0.0,
        "Custo de reposição": float(out["Custo de reposição"].sum()),
        "Preço sugerido (de)": np.nan,
        "Preço sugerido (até)": np.nan,
        "Estratégia de preço": "",
    }])
    out = pd.concat([out, total_row], ignore_index=True)

    return out, direction, float(min_cost)

def df_to_excel_bytes(sheets: dict) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, index=False, sheet_name=name[:31])
    bio.seek(0)
    return bio.getvalue()

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

@st.cache_data(show_spinner=False)
def read_report_workbook(file) -> dict:
    """
    Lê as abas do relatório gerado (kits_gerados...xlsx).
    Retorna dict: {sheet_name: df}
    """
    xl = pd.ExcelFile(file)
    out = {}
    for sh in xl.sheet_names:
        out[sh] = xl.parse(sh)
    return out

# -----------------------------
# SIDEBAR (inputs)
# -----------------------------
with st.sidebar:
    st.header("Entradas")
    base_file = st.file_uploader("Base de produtos (xlsx)", type=["xlsx"], key="base")
    report_file = st.file_uploader("Relatório de kits gerados (xlsx) [opcional]", type=["xlsx"], key="report")

    st.divider()
    st.header("Faixa do Kit")
    target_min = st.number_input("Preço mínimo do kit", value=TARGET_MIN_DEFAULT, step=10)
    target_max = st.number_input("Preço máximo do kit", value=TARGET_MAX_DEFAULT, step=10)

    st.divider()
    st.header("Simulador de compra")
    target_kits = st.number_input("Quantidade de torres", min_value=1, value=5, step=1)

# -----------------------------
# MAIN
# -----------------------------
if not base_file:
    st.info("Envie a base de produtos para iniciar.")
    st.stop()

base_df = load_base(base_file)

# Header style "painel"
kits_max, gargalo, cap_table = kits_possible_overall_correct(base_df)

top_left, top_mid, top_right = st.columns([2.4, 1.2, 0.8])
with top_left:
    st.markdown("<h1 style='margin:0;'>PAINEL DE TORRES</h1>", unsafe_allow_html=True)
    st.caption(f"Gargalo(s): {gargalo} | Faixa alvo: R$ {float(target_min):.2f} a R$ {float(target_max):.2f}")
with top_mid:
    st.metric("Quantidade de torres atualmente", kits_max)
with top_right:
    st.write("")

st.markdown("<hr/>", unsafe_allow_html=True)

# Abas (simulador + relatórios)
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Simulador de compra",
    "Kits resumo",
    "Kits itens",
    "Estoque restante",
    "Falha próximo kit"
])

# -----------------------------
# TAB 1 - Simulador
# -----------------------------
with tab1:
    left, right = st.columns([1.1, 2.9])

    with left:
        st.subheader("Quantidade de Torres")
        st.slider(" ", min_value=1, max_value=500, value=int(target_kits), step=1, key="kits_slider")
        # sincroniza number_input e slider (streamlit não tem 2-way perfeito; usamos o slider como fonte)
        target_kits_live = int(st.session_state.get("kits_slider", target_kits))

        st.caption("Dica: aumente/diminua e a tabela recalcula automaticamente.")

        # Export do simulador
        sim_table, direction, min_cost = simulator_purchase_table(
            base_df,
            target_kits_live,
            float(target_min),
            float(target_max)
        )

        sim_xlsx = df_to_excel_bytes({"simulador_compra": sim_table})
        st.download_button("Baixar Simulador (Excel)", data=sim_xlsx, file_name="simulador_compra.xlsx")
        st.download_button("Baixar Simulador (CSV)", data=df_to_csv_bytes(sim_table), file_name="simulador_compra.csv")

    with right:
        st.subheader("Simulador de compra")
        sim_table, direction, min_cost = simulator_purchase_table(
            base_df,
            target_kits_live,
            float(target_min),
            float(target_max)
        )

        # formatação visual
        display_df = sim_table.copy()
        # formata % e moeda
        display_df["Índice faltante por grupo"] = display_df["Índice faltante por grupo"].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "")
        display_df["Custo de reposição"] = display_df["Custo de reposição"].apply(lambda x: f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") if pd.notna(x) else "")
        display_df["Preço sugerido (de)"] = display_df["Preço sugerido (de)"].apply(lambda x: f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") if pd.notna(x) else "")
        display_df["Preço sugerido (até)"] = display_df["Preço sugerido (até)"].apply(lambda x: f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") if pd.notna(x) else "")

        st.dataframe(display_df, use_container_width=True, hide_index=True)

# -----------------------------
# Relatórios (abas 2-5)
# -----------------------------
report_dfs = {}
if report_file:
    try:
        report_dfs = read_report_workbook(report_file)
    except Exception as e:
        st.error(f"Erro ao ler relatório: {e}")

def render_report_tab(tab, sheet_name: str, title: str):
    with tab:
        st.subheader(title)

        if not report_file:
            st.info("Envie o arquivo de relatório (xlsx) na lateral para visualizar esta aba.")
            return

        if sheet_name not in report_dfs:
            st.warning(f"A aba '{sheet_name}' não foi encontrada nesse arquivo.")
            st.write(f"Abas encontradas: {list(report_dfs.keys())}")
            return

        df = report_dfs[sheet_name].copy()

        # downloads
        col_a, col_b, col_c = st.columns([1, 1, 2])
        with col_a:
            st.download_button(
                "Baixar Excel",
                data=df_to_excel_bytes({sheet_name: df}),
                file_name=f"{sheet_name}.xlsx"
            )
        with col_b:
            st.download_button(
                "Baixar CSV",
                data=df_to_csv_bytes(df),
                file_name=f"{sheet_name}.csv"
            )

        st.dataframe(df, use_container_width=True)

render_report_tab(tab2, "kits_resumo", "Kits resumo")
render_report_tab(tab3, "kits_itens", "Kits itens")
render_report_tab(tab4, "estoque_restante", "Estoque restante")
render_report_tab(tab5, "falha_proximo_kit", "Falha próximo kit")
