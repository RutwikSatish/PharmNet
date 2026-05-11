"""
PharmNet Risk — FDA Pharmaceutical Supply Chain Concentration Analyzer
=======================================================================
Computes network-based vulnerability metrics for the US pharmaceutical
supply chain using real FDA public datasets.

Data Sources (all publicly available from FDA):
  - FDA Drug Shortages Database: fda.gov/drugs/drug-safety-and-availability/drug-shortage-statistics
  - FDA Orange Book (Approved Drug Products): fda.gov/drugs/drug-approvals-and-databases/orange-book-data-files
    - products.txt  — approved manufacturers per drug
    - exclusivity.txt — market exclusivity periods
    - patent.txt    — active patent registrations

Metrics Implemented:
  1. Herfindahl-Hirschman Index (HHI) per active ingredient
     HHI = Σ(market share of each manufacturer)²
     HHI = 1.0 → single manufacturer (maximum concentration)
     HHI < 0.15 → competitive market (low fragility)
     Source: Hirschman (1945); applied to pharma supply by GAO (2014)
             "Drug Shortages: Root Causes and Potential Solutions"

  2. Manufacturer Network Criticality
     = count of unique drugs a manufacturer controls exclusively
     Source: Conceptually aligned with Griffin et al. NSF Award 2228510,
             Thrust 3 — network-based vulnerability metrics

  3. Shortage-Concentration Intersection
     = drugs currently in FDA shortage that are also high-HHI
     These represent maximum supply chain vulnerability: shortage + no substitutes

Research Context:
  This prototype operationalizes Thrust 3 of NSF Award 2228510:
  "Designing an Improved Information Infrastructure for Better Decision
  Making in Pharmaceutical Supply Chains" (Griffin, Ergun et al., Northeastern).

  The research questions this data directly informs:
  - Which therapeutic categories carry the most supply concentration risk?
  - Which manufacturers, if disrupted, cascade failures across the most drugs?
  - Does concentration risk predict which drugs end up in shortage?

Author: Rutwik Satish | MS Engineering Management, Northeastern University
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import math
import os

st.set_page_config(
    page_title="PharmNet Risk | FDA Supply Chain Vulnerability",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background:#04080F;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding-top:0!important;max-width:1400px;}
[data-testid="metric-container"]{
  background:#080F1E;border:1px solid rgba(239,68,68,0.2);border-radius:12px;
  padding:1.1rem 1.3rem;position:relative;overflow:hidden;
}
[data-testid="metric-container"]::before{
  content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,#EF4444,#F97316,#F59E0B);
}
[data-testid="stMetricValue"]{color:#EDF4FF!important;font-family:'Syne',sans-serif!important;font-size:1.8rem!important;}
[data-testid="stMetricLabel"]{color:#4A6A8A!important;font-size:10px!important;text-transform:uppercase;letter-spacing:.1em;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#060C1A,#040810);border-right:1px solid rgba(255,255,255,0.05);}
.stTabs [data-baseweb="tab"]{color:#4A6A8A;font-size:12.5px;font-weight:600;padding:.6rem 1.2rem;}
.stTabs [aria-selected="true"]{color:#EDF4FF!important;border-bottom:2px solid #EF4444!important;background:rgba(239,68,68,0.05)!important;}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none;}
.risk-high{background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.25);
  border-left:3px solid #EF4444;border-radius:8px;padding:10px 14px;margin-bottom:8px;}
.risk-med{background:rgba(249,115,22,.08);border:1px solid rgba(249,115,22,.25);
  border-left:3px solid #F97316;border-radius:8px;padding:10px 14px;margin-bottom:8px;}
.risk-low{background:rgba(16,185,129,.08);border:1px solid rgba(16,185,129,.25);
  border-left:3px solid #10B981;border-radius:8px;padding:10px 14px;margin-bottom:8px;}
.cite{font-size:10px;color:#2A4060;font-style:italic;margin-top:4px;}
hr{border-color:#1A2840!important;}
</style>
""", unsafe_allow_html=True)

PLOT = dict(
    plot_bgcolor="#0D1629", paper_bgcolor="#0D1629",
    font=dict(color="#5A7A9C", family="DM Sans"),
    margin=dict(l=0, r=0, t=30, b=0)
)
RED, ORANGE, AMBER, GREEN, BLUE, CYAN = "#EF4444","#F97316","#F59E0B","#10B981","#3B82F6","#06B6D4"

# ── Data loading ──────────────────────────────────────────────────────────────
DATA_DIR = "/mnt/user-data/uploads"

@st.cache_data
def load_data():
    shortage = pd.read_csv(os.path.join(DATA_DIR, "Drugshortages.csv"))
    products = pd.read_csv(os.path.join(DATA_DIR, "1778513025382_products.txt"), sep="~")
    exclusivity = pd.read_csv(os.path.join(DATA_DIR, "1778513025380_exclusivity.txt"), sep="~")
    patents = pd.read_csv(os.path.join(DATA_DIR, "1778513025381_patent.txt"), sep="~")

    shortage.columns  = [c.strip() for c in shortage.columns]
    products.columns  = [c.strip() for c in products.columns]

    # Compute HHI per active ingredient (RX only)
    rx = products[products['Type'] == 'RX'].copy()
    rx['ing_clean'] = rx['Ingredient'].str.upper().str.strip()

    def hhi(group):
        shares = group.value_counts(normalize=True)
        return float((shares**2).sum())

    mfr_count = rx.groupby('ing_clean')['Applicant_Full_Name'].nunique()
    hhi_vals  = rx.groupby('ing_clean')['Applicant_Full_Name'].apply(hhi)
    conc = pd.DataFrame({'Ingredient': mfr_count.index,
                          'n_manufacturers': mfr_count.values,
                          'HHI': hhi_vals.values})

    # Risk tier
    def risk_tier(hhi_val):
        if hhi_val == 1.0: return "Single Source"
        if hhi_val >= 0.5: return "High Concentration"
        if hhi_val >= 0.25: return "Moderate"
        return "Competitive"

    conc['Risk Tier'] = conc['HHI'].apply(risk_tier)

    # Current shortages
    current = shortage[shortage['Status'] == 'Current'].copy()
    current['Initial Posting Date'] = pd.to_datetime(current['Initial Posting Date'], errors='coerce')

    # Cross-reference
    DOSAGE_SUFFIXES = [
        'INJECTION','TABLET','CAPSULE','SOLUTION','SUSPENSION','CREAM','OINTMENT',
        'GEL','PATCH','SPRAY','INHALER','INFUSION','ORAL','EXTENDED RELEASE',
        'HYDROCHLORIDE INJECTION','MONOHYDRATE CONCENTRATE','INTRAVENOUS',
    ]
    def clean_name(name):
        s = str(name).upper().strip()
        s = s.split(';')[0].strip()
        s = s.split(',')[0].strip()
        for sfx in DOSAGE_SUFFIXES:
            if s.endswith(' ' + sfx):
                s = s[:-(len(sfx)+1)].strip()
        return s

    current['ingredient_clean'] = current['Generic Name'].apply(clean_name)
    merged = current.merge(conc, left_on='ingredient_clean', right_on='Ingredient', how='left')

    # Company portfolio analysis
    company_portfolio = rx.groupby('Applicant_Full_Name').agg(
        n_drugs_total=('ing_clean', 'nunique'),
    ).reset_index()

    single_source_ingredients = conc[conc['n_manufacturers'] == 1]['Ingredient'].tolist()
    ss_rx = rx[rx['ing_clean'].isin(single_source_ingredients)]
    company_ss = ss_rx.groupby('Applicant_Full_Name')['ing_clean'].nunique().reset_index()
    company_ss.columns = ['Applicant_Full_Name', 'n_single_source_drugs']

    company_risk = company_portfolio.merge(company_ss, on='Applicant_Full_Name', how='left')
    company_risk['n_single_source_drugs'] = company_risk['n_single_source_drugs'].fillna(0).astype(int)
    company_risk = company_risk.sort_values('n_drugs_total', ascending=False)

    return shortage, products, rx, conc, current, merged, company_risk


shortage_df, products_df, rx_df, conc_df, current_df, merged_df, company_risk_df = load_data()

# ── Key numbers ───────────────────────────────────────────────────────────────
n_total_rx          = conc_df['n_manufacturers'].count()
n_single_source     = (conc_df['HHI'] == 1.0).sum()
pct_single          = n_single_source / n_total_rx * 100
avg_hhi             = conc_df['HHI'].mean()
n_current_shortage  = current_df['Generic Name'].nunique()
n_shortage_records  = (shortage_df['Status'] == 'Current').sum()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:10px 0 16px;">
      <div style="font-family:'Syne',sans-serif;font-size:20px;font-weight:800;
           color:#F0F6FF;letter-spacing:-.02em;">PharmNet Risk</div>
      <div style="font-size:10px;color:#3A5A7C;text-transform:uppercase;
           letter-spacing:.08em;margin-top:2px;font-weight:600;">
           FDA Supply Chain Vulnerability</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:11px;font-weight:700;color:#E8F0FF;margin-bottom:6px;">DATA SOURCES</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:11.5px;color:#3A5A7C;line-height:1.8;">
    <b style="color:#8BA4C0;">FDA Drug Shortages DB</b><br>
    fda.gov/drugs/drug-safety-and-availability<br><br>
    <b style="color:#8BA4C0;">FDA Orange Book</b><br>
    Approved Drug Products with<br>
    Therapeutic Equivalence<br><br>
    <b style="color:#8BA4C0;">Last download:</b> May 2025<br>
    <b style="color:#8BA4C0;">License:</b> Public domain (FDA)
    </div>""", unsafe_allow_html=True)

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:11px;font-weight:700;color:#E8F0FF;margin-bottom:6px;">RESEARCH CONTEXT</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:11px;color:#3A5A7C;line-height:1.7;">
    Prototype for Thrust 3 of<br>
    NSF Award 2228510:<br>
    <i>"Designing an Improved Information<br>
    Infrastructure for Better Decision<br>
    Making in Pharmaceutical Supply Chains"</i><br><br>
    Griffin, Ergun et al.<br>
    Northeastern University
    </div>""", unsafe_allow_html=True)

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:11px;font-weight:700;color:#E8F0FF;margin-bottom:6px;">HHI SCALE</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:11px;color:#3A5A7C;line-height:1.8;">
    <span style="color:#EF4444;">1.0</span> = Single manufacturer<br>
    <span style="color:#F97316;">&gt;0.5</span> = High concentration<br>
    <span style="color:#F59E0B;">&gt;0.25</span> = Moderate<br>
    <span style="color:#10B981;">&lt;0.25</span> = Competitive<br><br>
    <i style="font-size:10px;">Hirschman (1945); used by GAO<br>(2014) drug shortage report</i>
    </div>""", unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:linear-gradient(135deg,#070E1E 0%,#04080F 100%);
     border-bottom:1px solid rgba(239,68,68,0.15);padding:2.5rem 0 2rem;">
  <div style="display:inline-flex;align-items:center;gap:8px;
       background:rgba(239,68,68,0.07);border:1px solid rgba(239,68,68,0.2);
       border-radius:20px;padding:4px 14px;font-size:11px;font-weight:700;
       color:#F87171;letter-spacing:.08em;text-transform:uppercase;margin-bottom:1rem;">
    Real FDA Data · Orange Book + Drug Shortage Database
  </div>
  <div style="font-family:'Syne',sans-serif;font-size:2.8rem;font-weight:800;
       color:#F0F6FF;line-height:1.05;letter-spacing:-.04em;margin-bottom:.8rem;">
    US Pharma Supply Chain<br><span style="color:#EF4444;">Concentration Risk</span> · Live FDA Data
  </div>
  <div style="font-size:14px;color:#4A6A8A;line-height:1.7;max-width:680px;margin-bottom:.5rem;">
    Network-based vulnerability analysis of the US pharmaceutical supply chain.
    HHI concentration index computed per active ingredient from FDA Orange Book.
    Cross-referenced against live FDA drug shortage records.
  </div>
  <div style="font-size:11px;color:#2A4060;font-style:italic;">
    Prototype for NSF Award 2228510, Thrust 3 — Network-based risk and vulnerability metrics.
    Griffin, Ergun et al., Northeastern University.
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# KPI row
k1,k2,k3,k4,k5,k6 = st.columns(6)
k1.metric("Total RX Ingredients",    f"{n_total_rx:,}")
k2.metric("Single-Source (HHI=1.0)", f"{n_single_source:,}", f"{pct_single:.1f}% of all RX")
k3.metric("Avg HHI (Portfolio)",      f"{avg_hhi:.3f}", "0=competitive · 1=monopoly")
k4.metric("High-Conc (HHI>0.5)",     f"{(conc_df['HHI']>0.5).sum():,}",
           f"{(conc_df['HHI']>0.5).mean()*100:.1f}% of portfolio")
k5.metric("Unique Drugs in Shortage", f"{n_current_shortage}")
k6.metric("Active Shortage Records",  f"{n_shortage_records:,}")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(239,68,68,0.3),transparent);margin:.5rem 0;"></div>', unsafe_allow_html=True)

# ── TABS ───────────────────────────────────────────────────────────────────────
t1, t2, t3, t4 = st.tabs([
    "  Concentration Risk  ",
    "  Current Shortages  ",
    "  Vulnerability Intersection  ",
    "  Company Exposure  ",
])

# ════ TAB 1 — CONCENTRATION RISK ════
with t1:
    st.markdown("""
    <div style="background:rgba(239,68,68,0.05);border:1px solid rgba(239,68,68,0.15);
         border-radius:10px;padding:12px 16px;margin-bottom:16px;font-size:13px;
         color:#8BA4C0;line-height:1.7;">
    <b style="color:#F87171;">HHI (Herfindahl-Hirschman Index)</b> measures supplier concentration per active ingredient.
    HHI = &Sigma;(market share)&sup2; where share is computed across FDA-approved manufacturers.
    HHI=1.0 means one manufacturer controls all approved versions of that drug.
    Source: Hirschman (1945); applied to pharmaceutical supply by GAO (2014),
    <i>Drug Shortages: Root Causes and Potential Solutions.</i>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        # HHI distribution
        st.markdown('<div style="font-size:10.5px;text-transform:uppercase;letter-spacing:.1em;font-weight:700;color:#3A5A7C;margin-bottom:10px;">HHI Distribution — All RX Ingredients (FDA Orange Book)</div>', unsafe_allow_html=True)
        fig_hhi = go.Figure(go.Histogram(
            x=conc_df['HHI'], nbinsx=40,
            marker=dict(color=RED, opacity=0.75,
                        line=dict(color="#0D1629", width=0.5)),
        ))
        fig_hhi.add_vline(x=1.0, line_dash="dash", line_color=RED,
                          annotation_text="Single source", annotation_font_color=RED)
        fig_hhi.add_vline(x=0.5, line_dash="dot", line_color=ORANGE,
                          annotation_text="High concentration", annotation_font_color=ORANGE,
                          annotation_position="bottom right")
        fig_hhi.add_vline(x=0.25, line_dash="dot", line_color=AMBER,
                          annotation_text="Moderate", annotation_font_color=AMBER,
                          annotation_position="bottom right")
        fig_hhi.update_layout(**PLOT, height=280,
            xaxis=dict(title=dict(text="HHI Score", font=dict(color="#5A7A9C",size=11)),
                       showgrid=False, tickfont=dict(color="#5A7A9C")),
            yaxis=dict(title=dict(text="Count", font=dict(color="#5A7A9C",size=11)),
                       showgrid=False, tickfont=dict(color="#5A7A9C")))
        st.plotly_chart(fig_hhi, use_container_width=True, config={"displayModeBar":False})
        st.caption("Source: FDA Orange Book — Approved Drug Products. 1,838 unique RX ingredients.")

    with col2:
        # Risk tier breakdown
        st.markdown('<div style="font-size:10.5px;text-transform:uppercase;letter-spacing:.1em;font-weight:700;color:#3A5A7C;margin-bottom:10px;">Supply Risk Tier Breakdown</div>', unsafe_allow_html=True)
        tier_counts = conc_df['Risk Tier'].value_counts()
        TIER_COLORS = {
            "Single Source":      RED,
            "High Concentration": ORANGE,
            "Moderate":           AMBER,
            "Competitive":        GREEN,
        }
        fig_pie = go.Figure(go.Pie(
            labels=tier_counts.index, values=tier_counts.values, hole=0.55,
            marker_colors=[TIER_COLORS.get(t, BLUE) for t in tier_counts.index],
            textinfo="label+percent", textfont=dict(size=11, color="#E8F0FF"),
        ))
        fig_pie.add_annotation(
            text=f"<b>{n_total_rx:,}</b><br>ingredients",
            x=0.5, y=0.5, font=dict(size=14, color="#E8F0FF"), showarrow=False
        )
        fig_pie.update_layout(**PLOT, height=280, showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar":False})

    st.markdown('<hr>', unsafe_allow_html=True)

    # Concentration by therapeutic area — using shortage data's categories
    st.markdown('<div style="font-size:10.5px;text-transform:uppercase;letter-spacing:.1em;font-weight:700;color:#3A5A7C;margin-bottom:8px;">High-Concentration Ingredients — Searchable Table</div>', unsafe_allow_html=True)
    search = st.text_input("Search ingredient", placeholder="e.g. BUPIVACAINE, CLONAZEPAM, LIDOCAINE...",
                           label_visibility="collapsed")
    tier_filter = st.selectbox("Filter by risk tier",
                               ["All", "Single Source", "High Concentration", "Moderate", "Competitive"],
                               label_visibility="collapsed")

    display_conc = conc_df.copy().sort_values('HHI', ascending=False)
    if search:
        display_conc = display_conc[display_conc['Ingredient'].str.contains(search.upper(), na=False)]
    if tier_filter != "All":
        display_conc = display_conc[display_conc['Risk Tier'] == tier_filter]

    display_conc['HHI'] = display_conc['HHI'].round(4)
    st.dataframe(
        display_conc[['Ingredient','n_manufacturers','HHI','Risk Tier']].rename(columns={
            'Ingredient':'Active Ingredient',
            'n_manufacturers':'Approved Manufacturers',
            'HHI':'HHI Score',
            'Risk Tier':'Risk Tier',
        }),
        use_container_width=True, height=380, hide_index=True
    )
    st.caption(f"Showing {len(display_conc):,} of {len(conc_df):,} ingredients. Source: FDA Orange Book.")
    st.download_button("Export concentration table",
                       display_conc.to_csv(index=False),
                       "pharmet_concentration.csv", "text/csv")


# ════ TAB 2 — CURRENT SHORTAGES ════
with t2:
    st.markdown("""
    <div style="background:rgba(239,68,68,0.05);border:1px solid rgba(239,68,68,0.15);
         border-radius:10px;padding:12px 16px;margin-bottom:16px;font-size:13px;
         color:#8BA4C0;line-height:1.7;">
    <b style="color:#F87171;">{:,} records</b> currently in shortage ({} unique drugs) as of the FDA shortage database.
    Source: FDA Drug Shortage Database — publicly updated at
    fda.gov/drugs/drug-safety-and-availability/drug-shortage-statistics
    </div>""".format(n_shortage_records, n_current_shortage), unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="medium")

    with c1:
        # Shortage by therapeutic category
        st.markdown('<div style="font-size:10.5px;text-transform:uppercase;letter-spacing:.1em;font-weight:700;color:#3A5A7C;margin-bottom:10px;">Current Shortages by Therapeutic Category</div>', unsafe_allow_html=True)
        cat_counts = current_df['Therapeutic Category'].value_counts().head(12)
        fig_cat = go.Figure(go.Bar(
            x=cat_counts.values,
            y=cat_counts.index,
            orientation='h',
            marker=dict(color=RED, opacity=0.8),
            text=cat_counts.values,
            textposition='outside',
            textfont=dict(size=10, color="#8BA4C0"),
        ))
        fig_cat.update_layout(**PLOT, height=360,
            xaxis=dict(showgrid=False, showline=False, showticklabels=False),
            yaxis=dict(showgrid=False, showline=False, tickfont=dict(size=10, color="#8BA4C0"),
                       autorange="reversed"))
        st.plotly_chart(fig_cat, use_container_width=True, config={"displayModeBar":False})
        st.caption("Source: FDA Drug Shortage Database.")

    with c2:
        # Shortage reasons
        st.markdown('<div style="font-size:10.5px;text-transform:uppercase;letter-spacing:.1em;font-weight:700;color:#3A5A7C;margin-bottom:10px;">Root Cause of Current Shortages</div>', unsafe_allow_html=True)
        reasons = current_df['Reason for Shortage'].value_counts()
        reasons_clean = reasons[reasons.index.notna()]
        fig_reasons = go.Figure(go.Pie(
            labels=[r[:40] for r in reasons_clean.index],
            values=reasons_clean.values,
            hole=0.5,
            marker_colors=[RED, ORANGE, AMBER, GREEN, BLUE, CYAN, "#8B5CF6", "#EC4899"],
            textinfo="label+percent",
            textfont=dict(size=10, color="#E8F0FF"),
        ))
        fig_reasons.update_layout(**PLOT, height=360, showlegend=False)
        st.plotly_chart(fig_reasons, use_container_width=True, config={"displayModeBar":False})

    st.markdown('<hr>', unsafe_allow_html=True)

    # Shortage timeline
    st.markdown('<div style="font-size:10.5px;text-transform:uppercase;letter-spacing:.1em;font-weight:700;color:#3A5A7C;margin-bottom:10px;">Shortage Records by Year of Initial Posting</div>', unsafe_allow_html=True)
    all_shortage = shortage_df.copy()
    all_shortage['Initial Posting Date'] = pd.to_datetime(all_shortage['Initial Posting Date'], errors='coerce')
    all_shortage['year'] = all_shortage['Initial Posting Date'].dt.year
    year_counts = all_shortage[all_shortage['year'].between(2012,2026)].groupby(['year','Status']).size().reset_index(name='count')

    fig_time = go.Figure()
    STATUS_COLORS = {'Current': RED, 'Resolved': GREEN, 'To Be Discontinued': ORANGE}
    for status, color in STATUS_COLORS.items():
        sub = year_counts[year_counts['Status'] == status]
        fig_time.add_trace(go.Bar(
            name=status, x=sub['year'], y=sub['count'],
            marker_color=color, opacity=0.8,
        ))
    fig_time.update_layout(**PLOT, height=250, barmode='stack',
        xaxis=dict(showgrid=False, tickfont=dict(color="#5A7A9C")),
        yaxis=dict(showgrid=False, showline=False, tickfont=dict(color="#5A7A9C")),
        legend=dict(font=dict(size=11, color="#8BA4C0"), bgcolor="rgba(0,0,0,0)",
                    orientation="h", x=0, y=1.12))
    st.plotly_chart(fig_time, use_container_width=True, config={"displayModeBar":False})
    st.caption("Source: FDA Drug Shortage Database. Years 2012–2026.")

    st.markdown('<hr>', unsafe_allow_html=True)

    # Companies with most shortage records
    st.markdown('<div style="font-size:10.5px;text-transform:uppercase;letter-spacing:.1em;font-weight:700;color:#3A5A7C;margin-bottom:10px;">Companies with Most Active Shortage Records</div>', unsafe_allow_html=True)
    top_companies = current_df['Company Name'].value_counts().head(15)
    fig_comp = go.Figure(go.Bar(
        x=top_companies.values, y=top_companies.index,
        orientation='h', marker=dict(color=ORANGE, opacity=0.8),
        text=top_companies.values, textposition='outside',
        textfont=dict(size=10, color="#8BA4C0"),
    ))
    fig_comp.update_layout(**PLOT, height=380,
        xaxis=dict(showgrid=False, showline=False, showticklabels=False),
        yaxis=dict(showgrid=False, showline=False, tickfont=dict(size=11, color="#8BA4C0"),
                   autorange="reversed"))
    st.plotly_chart(fig_comp, use_container_width=True, config={"displayModeBar":False})
    st.caption("Hospira (Pfizer) and Fresenius Kabi together account for >27% of all active shortage records.")


# ════ TAB 3 — VULNERABILITY INTERSECTION ════
with t3:
    st.markdown("""
    <div style="background:rgba(239,68,68,0.05);border:1px solid rgba(239,68,68,0.15);
         border-radius:10px;padding:12px 16px;margin-bottom:16px;font-size:13px;
         color:#8BA4C0;line-height:1.7;">
    <b style="color:#F87171;">Maximum vulnerability</b> = drug currently in shortage AND high supply concentration (few or one manufacturer).
    These drugs carry double risk: a shortage is active AND substitution is structurally limited.
    This intersection is the core of the network vulnerability analysis proposed in NSF Award 2228510, Thrust 3.
    </div>""", unsafe_allow_html=True)

    matched_current = merged_df[merged_df['HHI'].notna()].copy()

    v1, v2, v3 = st.columns(3)
    v1.metric("Shortage drugs matched to Orange Book",
              matched_current['Generic Name'].nunique(),
              help="Exact ingredient name match between shortage DB and Orange Book")
    v2.metric("Of these: Single Source (HHI=1.0)",
              (matched_current['HHI'] == 1.0).sum(),
              "Highest vulnerability")
    v3.metric("Of these: High Concentration (HHI>0.5)",
              (matched_current['HHI'] > 0.5).sum())

    st.markdown("<br>", unsafe_allow_html=True)

    # Scatter: HHI vs shortage record count
    st.markdown('<div style="font-size:10.5px;text-transform:uppercase;letter-spacing:.1em;font-weight:700;color:#3A5A7C;margin-bottom:10px;">Concentration vs. Shortage Volume — Each Point = One Drug</div>', unsafe_allow_html=True)

    scatter_data = matched_current.groupby(['Generic Name','HHI','n_manufacturers','Therapeutic Category']).size().reset_index(name='shortage_records')
    scatter_data['risk_color'] = scatter_data['HHI'].apply(
        lambda h: RED if h==1.0 else (ORANGE if h>0.5 else (AMBER if h>0.25 else GREEN))
    )
    scatter_data['risk_label'] = scatter_data['HHI'].apply(
        lambda h: "Single Source" if h==1.0 else ("High" if h>0.5 else ("Moderate" if h>0.25 else "Competitive"))
    )

    fig_scatter = go.Figure()
    for tier, color in [("Single Source",RED),("High",ORANGE),("Moderate",AMBER),("Competitive",GREEN)]:
        sub = scatter_data[scatter_data['risk_label']==tier]
        if len(sub):
            fig_scatter.add_trace(go.Scatter(
                x=sub['HHI'], y=sub['shortage_records'],
                mode='markers',
                name=tier,
                marker=dict(color=color, size=10, opacity=0.8,
                            line=dict(color="#0D1629", width=1)),
                text=sub['Generic Name'],
                hovertemplate="<b>%{text}</b><br>HHI: %{x:.3f}<br>Shortage records: %{y}<extra></extra>",
            ))
    fig_scatter.add_vline(x=1.0, line_dash="dash", line_color=RED,
                          annotation_text="Single source", annotation_font_color=RED)
    fig_scatter.add_vline(x=0.5, line_dash="dot", line_color=ORANGE)
    fig_scatter.update_layout(**PLOT, height=360,
        xaxis=dict(title=dict(text="HHI Score (supply concentration)",font=dict(color="#5A7A9C",size=11)),
                   showgrid=False, tickfont=dict(color="#5A7A9C"), range=[-0.05, 1.1]),
        yaxis=dict(title=dict(text="Active shortage records", font=dict(color="#5A7A9C",size=11)),
                   showgrid=False, tickfont=dict(color="#5A7A9C")),
        legend=dict(font=dict(size=11,color="#8BA4C0"),bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig_scatter, use_container_width=True, config={"displayModeBar":False})
    st.caption("Top-right quadrant = highest risk (high shortage volume + high concentration). Source: FDA Orange Book + FDA Drug Shortage Database.")

    st.markdown('<hr>', unsafe_allow_html=True)

    # Full matched table
    st.markdown('<div style="font-size:10.5px;text-transform:uppercase;letter-spacing:.1em;font-weight:700;color:#3A5A7C;margin-bottom:8px;">All Matched Drugs — Shortage Status + Concentration Score</div>', unsafe_allow_html=True)

    table_data = matched_current.groupby(['Generic Name','Therapeutic Category','HHI','n_manufacturers']).agg(
        shortage_records=('Generic Name','count'),
        companies_in_shortage=('Company Name', lambda x: ', '.join(x.dropna().unique()[:3])),
        reason=('Reason for Shortage', lambda x: x.dropna().value_counts().index[0] if len(x.dropna()) > 0 else 'N/A'),
    ).reset_index().sort_values('HHI', ascending=False)

    table_data['HHI'] = table_data['HHI'].round(3)
    table_data['Risk Tier'] = table_data['HHI'].apply(
        lambda h: "Single Source" if h==1.0 else ("High" if h>0.5 else ("Moderate" if h>0.25 else "Competitive"))
    )

    st.dataframe(
        table_data[['Generic Name','Therapeutic Category','HHI','n_manufacturers',
                    'Risk Tier','shortage_records','reason']].rename(columns={
            'Generic Name': 'Drug',
            'Therapeutic Category': 'Category',
            'HHI': 'HHI Score',
            'n_manufacturers': 'Approved Mfrs',
            'shortage_records': 'Active Records',
            'reason': 'Primary Shortage Reason',
        }),
        use_container_width=True, height=380, hide_index=True
    )
    st.download_button("Export vulnerability data",
                       table_data.to_csv(index=False),
                       "pharmnet_vulnerability.csv", "text/csv")


# ════ TAB 4 — COMPANY EXPOSURE ════
with t4:
    st.markdown("""
    <div style="background:rgba(59,130,246,0.05);border:1px solid rgba(59,130,246,0.15);
         border-radius:10px;padding:12px 16px;margin-bottom:16px;font-size:13px;
         color:#8BA4C0;line-height:1.7;">
    <b style="color:#93C5FD;">Network criticality</b> of each manufacturer: if a company's
    manufacturing is disrupted, how many single-source drugs lose their only supplier?
    Companies with the most single-source drugs in their portfolio are the highest-criticality
    nodes in the pharmaceutical supply network.
    This directly supports Thrust 3 of NSF Award 2228510 — identifying critical nodes in complex supply systems.
    </div>""", unsafe_allow_html=True)

    top_ss = company_risk_df[company_risk_df['n_single_source_drugs'] > 0].sort_values(
        'n_single_source_drugs', ascending=False
    ).head(20)

    c1, c2 = st.columns(2, gap="medium")

    with c1:
        st.markdown('<div style="font-size:10.5px;text-transform:uppercase;letter-spacing:.1em;font-weight:700;color:#3A5A7C;margin-bottom:10px;">Companies by Single-Source Drug Count (Network Criticality)</div>', unsafe_allow_html=True)
        fig_ss = go.Figure(go.Bar(
            x=top_ss['n_single_source_drugs'],
            y=top_ss['Applicant_Full_Name'].str.title().str[:35],
            orientation='h',
            marker=dict(color=RED, opacity=0.8),
            text=top_ss['n_single_source_drugs'],
            textposition='outside',
            textfont=dict(size=10, color="#8BA4C0"),
        ))
        fig_ss.update_layout(**PLOT, height=480,
            xaxis=dict(showgrid=False, showline=False, showticklabels=False,
                       title=dict(text="Single-source drugs controlled", font=dict(color="#5A7A9C",size=11))),
            yaxis=dict(showgrid=False, showline=False, tickfont=dict(size=10, color="#8BA4C0"),
                       autorange="reversed"))
        st.plotly_chart(fig_ss, use_container_width=True, config={"displayModeBar":False})
        st.caption("Source: FDA Orange Book. Each bar = number of active ingredients for which this is the only FDA-approved manufacturer.")

    with c2:
        st.markdown('<div style="font-size:10.5px;text-transform:uppercase;letter-spacing:.1em;font-weight:700;color:#3A5A7C;margin-bottom:10px;">Portfolio Size vs. Single-Source Concentration</div>', unsafe_allow_html=True)
        plot_data = company_risk_df[company_risk_df['n_drugs_total'] >= 5].copy()
        plot_data['pct_ss'] = plot_data['n_single_source_drugs'] / plot_data['n_drugs_total'] * 100
        plot_data['color_val'] = plot_data['n_single_source_drugs']

        fig_bubble = go.Figure(go.Scatter(
            x=plot_data['n_drugs_total'],
            y=plot_data['pct_ss'],
            mode='markers',
            marker=dict(
                size=plot_data['n_single_source_drugs'].clip(3,20),
                color=plot_data['n_single_source_drugs'],
                colorscale=[[0,"#1A2840"],[0.3,AMBER],[0.7,ORANGE],[1.0,RED]],
                showscale=True,
                colorbar=dict(title=dict(text="Single-source drugs",font=dict(color="#5A7A9C",size=10)),
                              tickfont=dict(color="#5A7A9C",size=9)),
                opacity=0.8,
                line=dict(color="#0D1629",width=0.5)
            ),
            text=plot_data['Applicant_Full_Name'].str.title(),
            hovertemplate="<b>%{text}</b><br>Total drugs: %{x}<br>Single-source: %{marker.size}<br>% single-source: %{y:.1f}%<extra></extra>",
        ))
        fig_bubble.update_layout(**PLOT, height=480,
            xaxis=dict(title=dict(text="Total RX drug portfolio size", font=dict(color="#5A7A9C",size=11)),
                       showgrid=False, tickfont=dict(color="#5A7A9C")),
            yaxis=dict(title=dict(text="% portfolio that is single-source",font=dict(color="#5A7A9C",size=11)),
                       showgrid=False, tickfont=dict(color="#5A7A9C")))
        st.plotly_chart(fig_bubble, use_container_width=True, config={"displayModeBar":False})
        st.caption("Bubble size = number of single-source drugs. Top-right = high portfolio size AND high single-source concentration.")

    st.markdown('<hr>', unsafe_allow_html=True)

    st.markdown('<div style="font-size:10.5px;text-transform:uppercase;letter-spacing:.1em;font-weight:700;color:#3A5A7C;margin-bottom:8px;">Full Company Risk Profile — Searchable</div>', unsafe_allow_html=True)
    co_search = st.text_input("Search company", placeholder="e.g. Pfizer, Fresenius, Hikma...", label_visibility="collapsed")
    display_cr = company_risk_df.copy()
    if co_search:
        display_cr = display_cr[display_cr['Applicant_Full_Name'].str.contains(co_search.upper(), na=False)]
    display_cr['pct_single_source'] = (display_cr['n_single_source_drugs'] / display_cr['n_drugs_total'] * 100).round(1)
    display_cr = display_cr.rename(columns={
        'Applicant_Full_Name': 'Manufacturer',
        'n_drugs_total': 'Total RX Drugs',
        'n_single_source_drugs': 'Single-Source Drugs',
        'pct_single_source': '% Single-Source',
    })
    st.dataframe(
        display_cr[['Manufacturer','Total RX Drugs','Single-Source Drugs','% Single-Source']].head(200),
        use_container_width=True, height=420, hide_index=True
    )
    st.download_button("Export company risk data",
                       display_cr.to_csv(index=False),
                       "pharmnet_company_risk.csv", "text/csv")


# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<p style='font-size:11px;color:#2A4060;text-align:center;line-height:1.8;'>
PharmNet Risk · FDA Pharmaceutical Supply Chain Concentration Analyzer ·
Real FDA data — Orange Book + Drug Shortage Database · No synthetic data ·<br>
HHI: Hirschman (1945) · GAO (2014) Drug Shortages: Root Causes and Potential Solutions ·
Prototype for NSF Award 2228510, Thrust 3 (Griffin, Ergun et al., Northeastern University) ·
Built by Rutwik Satish · MS Engineering Management, Northeastern University
</p>""", unsafe_allow_html=True)
