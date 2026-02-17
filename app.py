import re
import io
import pandas as pd
import numpy as np
import streamlit as st

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Dashboard de Torres/Kits", layout="wide")

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
ADJUST_CATS = {"BR_DEMAIS", "CO"}  # categorias de "ajuste fino" no valor do kit

# -----------------------------
# HELPERS
# -----------------------------
def norm_sku(s: str) -> str:
    return re.sub(r"\s+", "", str(s).strip()).upper()

def assign_category(row) -> str:
    sku = row["Sku_norm"]

    # Correntes por flags (prioridade)
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

    # colunas obrigatórias
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
    """
    Consolida por SKU_norm:
    - Estoque = máximo (assumindo que SKU_norm identifica o item)
    - Preco = mínimo (ou o preço "representativo" por SKU)
    """
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
# CAPACIDADE CORRETA (com repetição entre kits conforme estoque)
# -----------------------------
def max_kits_category_from_stocks(stocks: np.ndarray, m: int) -> int:
    """
    Máximo k tal que sum(min(stocks, k)) >= k*m
    (cada SKU pode aparecer no máximo 1x por kit, mas pode repetir em kits diferentes até acabar estoque)
    """
    stocks = np.array(stocks, dtype=int)
    stocks = stocks[stocks > 0]
    if len(stocks) < m:
        return 0

    hi = int(stocks.sum() // m)  # upper bound por unidades
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
    """
    Quantos 'usos' faltam para viabilizar k kits:
      falta = k*m - sum(min(stocks, k))
    """
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
    return out.sort_values("categoria")

def kits_possible_overall_correct(df: pd.DataFrame) -> tuple[int, str, pd.DataFrame]:
    t = capacity_table_correct(df)
    kits_max = int(t["kits_max_cat"].min()) if len(t) else 0
    gargalos = t.loc[t["kits_max_cat"] == kits_max, "categoria"].tolist()
    gargalo_str = ", ".join(gargalos) if gargalos else "-"
    return kits_max, gargalo_str, t

# -----------------------------
# HEURÍSTICA DE PREÇO "INTELIGENTE"
# -----------------------------
def min_cost_theoretical(bs: pd.DataFrame) -> float:
    """
    Custo mínimo teórico do kit:
    soma dos 'mínimos mais baratos' por categoria.
    """
    total = 0.0
    for cat, (mn, _) in RULES.items():
        prices = bs.loc[bs["categoria"] == cat, "Preco"].sort_values().to_numpy()
        if len(prices) < mn:
            return np.inf
        total += float(prices[:mn].sum())
    return float(total)

def choose_price_band(direction: str, weight: float, is_adjust: bool):
    """
    direction:
      - 'cheaper'  => mínimo teórico já está caro (precisa comprar mais barato)
      - 'pricier'  => mínimo teórico está baixo (pode comprar mais caro pra subir valor)
      - 'neutral'  => mínimo teórico já ok
    weight: quanto a categoria pesa no custo do kit (0..1)
    """
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

    # neutral
    if is_adjust:
        return ("p10", "p90", "ajuste amplo (P10–P90)")
    return ("p25", "p75", "faixa padrão (P25–P75)")

def replenishment_plan(df: pd.DataFrame, target_kits: int, target_min: float, target_max: float, sku_units_assumption: int):
    """
    Plano de reposição:
    - usa falta_slots (métrica correta) para viabilizar N kits por categoria
    - estima 'novos_skus' assumindo X unidades por SKU novo (default = 1 ou 2)
    """
    cat, bs = summarize_category(df)

    # direção geral baseada no custo mínimo teórico
    min_cost = min_cost_theoretical(bs)
    if np.isinf(min_cost):
        direction = "neutral"
    elif min_cost > target_max:
        direction = "cheaper"
    elif min_cost < target_min:
        direction = "pricier"
    else:
        direction = "neutral"

    # peso relativo: mediana * mínimo por kit
    cat["contrib"] = cat["preco_med"] * cat["min_por_kit"]
    contrib_sum = float(cat["contrib"].sum()) if float(cat["contrib"].sum()) > 0 else 1.0
    cat["peso"] = cat["contrib"] / contrib_sum

    # falta_slots por categoria (métrica correta)
    slots = []
    for _, r in cat.iterrows():
        c = r["categoria"]
        mn = int(r["min_por_kit"])
        stocks = bs.loc[bs["categoria"] == c, "Estoque"].to_numpy(dtype=int)
        slots.append(shortage_slots_for_target(stocks, mn, int(target_kits)))
    cat["falta_slots"] = pd.Series(slots, index=cat.index).astype(int)

    # sugestão de "novos SKUs" se cada SKU novo vier com sku_units_assumption unidades
    cat["novos_skus_est_1un"] = np.ceil(cat["falta_slots"] / 1.0).astype(int)
    cat["novos_skus_est_Xun"] = np.ceil(cat["falta_slots"] / float(max(1, sku_units_assumption))).astype(int)

    # faixa inteligente por categoria
    lo_list, hi_list, label_list = [], [], []
    for _, r in cat.iterrows():
        is_adjust = r["categoria"] in ADJUST_CATS
        lo, hi, label = choose_price_band(direction, float(r["peso"]), is_adjust)
        lo_list.append(lo); hi_list.append(hi); label_list.append(label)

    cat["estrategia_preco"] = label_list
    cat["preco_sugerido_de"] = [round(float(r[f"preco_{lo}"]), 2) for r, lo in zip(cat.to_dict("records"), lo_list)]
    cat["preco_sugerido_ate"] = [round(float(r[f"preco_{hi}"]), 2) for r, hi in zip(cat.to_dict("records"), hi_list)]

    out = cat[[
        "categoria",
        "min_por_kit", "max_por_kit",
        "skus_unicos", "estoque_total",
        "falta_slots",
        "novos_skus_est_1un",
        "novos_skus_est_Xun",
        "preco_sugerido_de", "preco_sugerido_ate",
        "estrategia_preco",
    ]].copy()

    out["max_por_kit"] = out["max_por_kit"].replace([np.inf], ["∞"])
    out["impacto"] = out["falta_slots"]  # principal métrica agora
    out = out.sort_values(["impacto", "categoria"], ascending=[False, True])

    return out, direction, min_cost

# -----------------------------
# UI
# -----------------------------
st.title("Dashboard de Torres/Kits")

with st.sidebar:
    st.header("Entrada")
    file = st.file_uploader("Envie sua base (Excel)", type=["xlsx"])
    target_min = st.number_input("Preço mínimo do kit", value=TARGET_MIN_DEFAULT, step=10)
    target_max = st.number_input("Preço máximo do kit", value=TARGET_MAX_DEFAULT, step=10)

    st.divider()
    st.header("Planejar reposição")
    target_kits = st.number_input("Quantos kits você quer montar?", min_value=1, value=100, step=1)
    sku_units_assumption = st.selectbox(
        "Suposição de compra: unidades por SKU novo",
        options=[1, 2, 3, 5],
        index=0
    )

if not file:
    st.info("Faça upload do Excel para começar.")
    st.stop()

df = load_base(file)
st.caption(f"Faixa alvo do kit: R$ {float(target_min):.2f} a R$ {float(target_max):.2f}")

col1, col2 = st.columns([1, 1])

# capacidade atual (CORRIGIDA)
kits_max, gargalo, cap_table = kits_possible_overall_correct(df)
with col1:
    st.subheader("Capacidade atual (correta)")
    st.metric("Kits possíveis hoje", kits_max)
    st.write(f"**Gargalo(s):** {gargalo}")

    # tabela por categoria
    view = cap_table.copy()
    st.dataframe(view[[
        "categoria",
        "min_por_kit", "max_por_kit",
        "skus_unicos", "estoque_total",
        "kits_max_cat",
        "preco_min", "preco_med", "preco_max",
        "gargalo"
    ]], use_container_width=True)

# plano de reposição inteligente (CORRIGIDO)
with col2:
    st.subheader("Plano de reposição (para atingir a meta)")
    plan, direction, min_cost = replenishment_plan(df, int(target_kits), float(target_min), float(target_max), int(sku_units_assumption))

    # diagnóstico direção do preço
    if np.isinf(min_cost):
        st.warning("Não foi possível calcular o custo mínimo teórico: alguma categoria não tem SKUs suficientes para os mínimos.")
    else:
        st.write(f"**Custo mínimo teórico (mínimos mais baratos):** R$ {min_cost:.2f}")
        if direction == "cheaper":
            st.error("Diagnóstico: mínimo teórico está acima do teto → **compras devem puxar mais para barato**.")
        elif direction == "pricier":
            st.info("Diagnóstico: mínimo teórico está abaixo do piso → **compras podem puxar mais para caro**.")
        else:
            st.success("Diagnóstico: mínimo teórico compatível → **faixas padrão + ajuste fino**.")

    st.dataframe(plan, use_container_width=True)

    faltas = plan[plan["falta_slots"] > 0]
    if len(faltas) == 0:
        st.success("Pelos cálculos de capacidade (com estoques), você já consegue atingir a meta (pelos mínimos).")
    else:
        st.warning(f"Há {len(faltas)} categorias com falta para atingir {int(target_kits)} kits (pelos mínimos).")

    # download do plano
    export_bytes = io.BytesIO()
    with pd.ExcelWriter(export_bytes, engine="openpyxl") as writer:
        plan.to_excel(writer, index=False, sheet_name="plano_reposicao")
        cap_table.to_excel(writer, index=False, sheet_name="capacidade_por_categoria")
    export_bytes.seek(0)

    st.download_button(
        "Baixar plano (Excel)",
        data=export_bytes,
        file_name="plano_reposicao_kits.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

st.divider()
st.subheader("Como ler (importante)")
st.write("""
### Por que agora bate com o seu gerador?
Antes, a estimativa travava em **1** porque usava `skus_unicos // min_por_kit`, assumindo (errado) que **cada SKU só pode aparecer em 1 kit**.

Agora a capacidade usa a condição correta:
- cada SKU pode aparecer **em vários kits**, até acabar estoque
- mas **no máximo 1 vez por kit**
- então a capacidade por categoria é o maior `k` tal que `sum(min(estoque_sku, k)) >= k * min_por_kit`

### Plano de reposição
- **falta_slots** = quantos “usos/unidades” faltam para aquela categoria sustentar `N` kits (respeitando 1x por kit por SKU)
- **novos_skus_est_1un** = se você comprar **1 unidade por SKU novo**, quantos SKUs novos precisa (heurística)
- **novos_skus_est_Xun** = mesma ideia, mas assumindo X unidades por SKU novo (configurável na lateral)
- Faixa de preço sugerida continua por percentis e “direção” (baratear/subir/neutral).
""")
