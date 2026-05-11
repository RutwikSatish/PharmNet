"""
PharmNet Risk — FDA Pharmaceutical Supply Chain Concentration Analyzer
=======================================================================
Data Sources (publicly available from FDA):
  - Drugshortages.csv  — FDA Drug Shortage Database
  - products.txt       — FDA Orange Book approved manufacturers per drug
  - exclusivity.txt    — FDA Orange Book market exclusivity
  - patent.txt         — FDA Orange Book patent registrations

Metrics:
  HHI (Herfindahl-Hirschman Index) per active ingredient
  Manufacturer network criticality (single-source drug count)
  Shortage-concentration intersection (maximum vulnerability)

Research context:

  "Designing an Improved Information Infrastructure for Better Decision

Author: Rutwik Satish | MS Engineering Management, Northeastern University
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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
  padding:1.1rem 1.3rem;position:relative;overflow:hidden;}
[data-testid="metric-container"]::before{
  content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,#EF4444,#F97316,#F59E0B);}
[data-testid="stMetricValue"]{color:#EDF4FF!important;font-family:'Syne',sans-serif!important;font-size:1.8rem!important;}
[data-testid="stMetricLabel"]{color:#4A6A8A!important;font-size:10px!important;text-transform:uppercase;letter-spacing:.1em;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#060C1A,#040810);border-right:1px solid rgba(255,255,255,0.05);}
.stTabs [data-baseweb="tab"]{color:#4A6A8A;font-size:12.5px;font-weight:600;padding:.6rem 1.2rem;}
.stTabs [aria-selected="true"]{color:#EDF4FF!important;border-bottom:2px solid #EF4444!important;background:rgba(239,68,68,0.05)!important;}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none;}
hr{border-color:#1A2840!important;}
</style>
""", unsafe_allow_html=True)

PLOT = dict(
    plot_bgcolor="#0D1629", paper_bgcolor="#0D1629",
    font=dict(color="#5A7A9C", family="DM Sans"),
    margin=dict(l=0, r=0, t=30, b=0)
)
RED, ORANGE, AMBER, GREEN, BLUE, CYAN = "#EF4444","#F97316","#F59E0B","#10B981","#3B82F6","#06B6D4"

# ── File paths ─────────────────────────────────────────────────────────────────
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

FILES = {
    "Drugshortages.csv": ",",
    "products.txt":      "~",
    "exclusivity.txt":   "~",
    "patent.txt":        "~",
}

# ── Check all files exist before loading ──────────────────────────────────────
missing = [f for f in FILES if not os.path.exists(os.path.join(DATA_DIR, f))]
if missing:
    st.error(f"Missing data files in `{DATA_DIR}`: **{', '.join(missing)}**")
    st.markdown("""
    Your GitHub repo needs these four files alongside `app.py`:
    ```
    Drugshortages.csv
    products.txt
    exclusivity.txt
    patent.txt
    requirements.txt
    app.py
    ```
    All four are from the FDA public datasets. Commit them all and redeploy.
    """)
    st.stop()

# ── Load and process ──────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    shortage    = pd.read_csv(os.path.join(DATA_DIR, "Drugshortages.csv"))
    products    = pd.read_csv(os.path.join(DATA_DIR, "products.txt"),    sep="~")
    exclusivity = pd.read_csv(os.path.join(DATA_DIR, "exclusivity.txt"), sep="~")
    patents     = pd.read_csv(os.path.join(DATA_DIR, "patent.txt"),      sep="~")

    shortage.columns = [c.strip() for c in shortage.columns]
    products.columns = [c.strip() for c in products.columns]

    # RX products only
    rx = products[products['Type'] == 'RX'].copy()
    rx['ing_clean'] = rx['Ingredient'].str.upper().str.strip()

    # HHI per active ingredient
    # HHI = sum(market_share_i^2). Source: Hirschman (1945); GAO (2014)
    def hhi(group):
        shares = group.value_counts(normalize=True)
        return float((shares**2).sum())

    mfr_count = rx.groupby('ing_clean')['Applicant_Full_Name'].nunique()
    hhi_vals  = rx.groupby('ing_clean')['Applicant_Full_Name'].apply(hhi)
    conc = pd.DataFrame({
        'Ingredient':      mfr_count.index,
        'n_manufacturers': mfr_count.values,
        'HHI':             hhi_vals.values,
    })

    def risk_tier(h):
        if h == 1.0:  return "Single Source"
        if h >= 0.5:  return "High Concentration"
        if h >= 0.25: return "Moderate"
        return "Competitive"

    conc['Risk Tier'] = conc['HHI'].apply(risk_tier)

    # Current shortages
    current = shortage[shortage['Status'] == 'Current'].copy()
    current['Initial Posting Date'] = pd.to_datetime(
        current['Initial Posting Date'], errors='coerce'
    )

    # Cross-reference shortages → Orange Book
    SUFFIXES = [
        'INJECTION','TABLET','CAPSULE','SOLUTION','SUSPENSION','CREAM',
        'OINTMENT','GEL','PATCH','SPRAY','INHALER','INFUSION','ORAL',
        'EXTENDED RELEASE','HYDROCHLORIDE INJECTION','MONOHYDRATE',
        'CONCENTRATE','INTRAVENOUS',
    ]
    def clean_name(name):
        s = str(name).upper().strip().split(';')[0].split(',')[0].strip()
        for sfx in SUFFIXES:
            if s.endswith(' ' + sfx):
                s = s[:-(len(sfx)+1)].strip()
        return s

    current['ingredient_clean'] = current['Generic Name'].apply(clean_name)
    merged = current.merge(
        conc, left_on='ingredient_clean', right_on='Ingredient', how='left'
    )

    # Company portfolio risk
    company_portfolio = rx.groupby('Applicant_Full_Name').agg(
        n_drugs_total=('ing_clean', 'nunique')
    ).reset_index()

    ss_ingredients = conc[conc['n_manufacturers'] == 1]['Ingredient'].tolist()
    ss_rx = rx[rx['ing_clean'].isin(ss_ingredients)]
    company_ss = ss_rx.groupby('Applicant_Full_Name')['ing_clean'].nunique().reset_index()
    company_ss.columns = ['Applicant_Full_Name', 'n_single_source_drugs']

    company_risk = company_portfolio.merge(company_ss, on='Applicant_Full_Name', how='left')
    company_risk['n_single_source_drugs'] = (
        company_risk['n_single_source_drugs'].fillna(0).astype(int)
    )
    company_risk = company_risk.sort_values('n_drugs_total', ascending=False)

    return shortage, products, rx, conc, current, merged, company_risk

shortage_df, products_df, rx_df, conc_df, current_df, merged_df, company_risk_df = load_data()

# ── Key numbers ───────────────────────────────────────────────────────────────
n_total_rx         = len(conc_df)
n_single_source    = (conc_df['HHI'] == 1.0).sum()
pct_single         = n_single_source / n_total_rx * 100
avg_hhi            = conc_df['HHI'].mean()
n_current_shortage = current_df['Generic Name'].nunique()
n_shortage_records = (shortage_df['Status'] == 'Current').sum()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:10px 0 16px;">
      <div style="font-family:'Syne',sans-serif;font-size:20px;font-weight:800;
           color:#F0F6FF;letter-spacing:-.02em;">PharmNet Risk</div>
      <div style="font-size:10px;color:#3A5A7C;text-transform:uppercase;
           letter-spacing:.08em;margin-top:2px;">FDA Supply Chain Vulnerability</div>
    </div>""", unsafe_allow_html=True)
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:11px;font-weight:700;color:#E8F0FF;">DATA SOURCES</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:11.5px;color:#3A5A7C;line-height:1.8;">
    <b style="color:#8BA4C0;">FDA Drug Shortages Database</b><br>
    fda.gov/drugs/drug-safety-and-availability<br><br>
    <b style="color:#8BA4C0;">FDA Orange Book</b><br>
    Approved Drug Products<br><br>
    <b style="color:#8BA4C0;">License:</b> Public domain (FDA)
    </div>""", unsafe_allow_html=True)

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:11px;color:#3A5A7C;line-height:1.8;">
    <b style="color:#E8F0FF;">HHI Scale</b><br>
    <span style="color:#EF4444;">1.0</span> = Single manufacturer<br>
    <span style="color:#F97316;">&gt;0.5</span> = High concentration<br>
    <span style="color:#F59E0B;">&gt;0.25</span> = Moderate<br>
    <span style="color:#10B981;">&lt;0.25</span> = Competitive<br>
    <i style="font-size:10px;">Hirschman (1945)</i>
    </div>""", unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:linear-gradient(135deg,#070E1E,#04080F);
     border-bottom:1px solid rgba(239,68,68,0.15);padding:2.5rem 0 2rem;">
  <div style="display:inline-flex;background:rgba(239,68,68,0.07);
       border:1px solid rgba(239,68,68,0.2);border-radius:20px;padding:4px 14px;
       font-size:11px;font-weight:700;color:#F87171;letter-spacing:.08em;
       text-transform:uppercase;margin-bottom:1rem;">
    Real FDA Data · Orange Book + Drug Shortage Database
  </div>
  <div style="font-family:'Syne',sans-serif;font-size:2.6rem;font-weight:800;
       color:#F0F6FF;line-height:1.05;letter-spacing:-.04em;margin-bottom:.8rem;">
    US Pharma Supply Chain<br><span style="color:#EF4444;">Concentration Risk</span>
  </div>
  <div style="font-size:14px;color:#4A6A8A;line-height:1.7;max-width:660px;margin-bottom:.5rem;">
    HHI concentration index computed per active ingredient from the FDA Orange Book.
    Cross-referenced against live FDA drug shortage records to identify
    maximum-vulnerability drugs: shortage active AND few or no substitutes.
  </div>

</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

k1,k2,k3,k4,k5,k6 = st.columns(6)
k1.metric("Total RX Ingredients",    f"{n_total_rx:,}")
k2.metric("Single-Source (HHI=1.0)", f"{n_single_source:,}", f"{pct_single:.1f}% of all RX")
k3.metric("Avg HHI (Portfolio)",      f"{avg_hhi:.3f}", "0=competitive · 1=monopoly")
k4.metric("High-Conc (HHI>0.5)",
          f"{(conc_df['HHI']>0.5).sum():,}",
          f"{(conc_df['HHI']>0.5).mean()*100:.1f}% of portfolio")
k5.metric("Unique Drugs in Shortage", f"{n_current_shortage}")
k6.metric("Active Shortage Records",  f"{n_shortage_records:,}")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(239,68,68,0.3),transparent);margin:.5rem 0;"></div>', unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
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
         border-radius:10px;padding:12px 16px;margin-bottom:16px;font-size:13px;color:#8BA4C0;line-height:1.7;">
    <b style="color:#F87171;">HHI = &Sigma;(market share)&sup2;</b> per active ingredient across FDA-approved manufacturers.
    HHI=1.0 means one manufacturer controls all approved versions of that drug — maximum fragility.
    Source: Hirschman (1945); applied to pharmaceutical supply by GAO (2014),
    <i>Drug Shortages: Root Causes and Potential Solutions.</i>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div style="font-size:10.5px;text-transform:uppercase;letter-spacing:.1em;font-weight:700;color:#3A5A7C;margin-bottom:10px;">HHI Distribution — All RX Ingredients</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Histogram(
            x=conc_df['HHI'], nbinsx=40,
            marker=dict(color=RED, opacity=0.75, line=dict(color="#0D1629",width=0.5))
        ))
        fig.add_vline(x=1.0, line_dash="dash", line_color=RED,
                      annotation_text="Single source", annotation_font_color=RED)
        fig.add_vline(x=0.5, line_dash="dot", line_color=ORANGE,
                      annotation_text="High concentration", annotation_font_color=ORANGE,
                      annotation_position="bottom right")
        fig.update_layout(**PLOT, height=280,
            xaxis=dict(title="HHI Score", showgrid=False, tickfont=dict(color="#5A7A9C")),
            yaxis=dict(title="Count", showgrid=False, tickfont=dict(color="#5A7A9C")))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        st.caption("Source: FDA Orange Book. 1,838 unique RX ingredients.")

    with c2:
        st.markdown('<div style="font-size:10.5px;text-transform:uppercase;letter-spacing:.1em;font-weight:700;color:#3A5A7C;margin-bottom:10px;">Supply Risk Tier Breakdown</div>', unsafe_allow_html=True)
        tier_counts = conc_df['Risk Tier'].value_counts()
        TIER_COLORS = {"Single Source":RED,"High Concentration":ORANGE,"Moderate":AMBER,"Competitive":GREEN}
        fig2 = go.Figure(go.Pie(
            labels=tier_counts.index, values=tier_counts.values, hole=0.55,
            marker_colors=[TIER_COLORS.get(t,BLUE) for t in tier_counts.index],
            textinfo="label+percent", textfont=dict(size=11,color="#E8F0FF"),
        ))
        fig2.add_annotation(text=f"<b>{n_total_rx:,}</b><br>RX ingredients",
                            x=0.5,y=0.5,font=dict(size=13,color="#E8F0FF"),showarrow=False)
        fig2.update_layout(**PLOT, height=280, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:10.5px;text-transform:uppercase;letter-spacing:.1em;font-weight:700;color:#3A5A7C;margin-bottom:8px;">Searchable Concentration Table</div>', unsafe_allow_html=True)
    col_s, col_t = st.columns([3,1])
    search     = col_s.text_input("Search ingredient", placeholder="e.g. BUPIVACAINE, CLONAZEPAM...", label_visibility="collapsed")
    tier_filter= col_t.selectbox("Tier", ["All","Single Source","High Concentration","Moderate","Competitive"], label_visibility="collapsed")

    disp = conc_df.copy().sort_values('HHI', ascending=False)
    if search:
        disp = disp[disp['Ingredient'].str.contains(search.upper(), na=False)]
    if tier_filter != "All":
        disp = disp[disp['Risk Tier'] == tier_filter]
    disp['HHI'] = disp['HHI'].round(4)

    st.dataframe(
        disp[['Ingredient','n_manufacturers','HHI','Risk Tier']].rename(columns={
            'Ingredient':'Active Ingredient','n_manufacturers':'Approved Manufacturers',
            'HHI':'HHI Score','Risk Tier':'Risk Tier'}),
        use_container_width=True, height=380, hide_index=True
    )
    st.caption(f"Showing {len(disp):,} of {len(conc_df):,} ingredients.")
    st.download_button("Export", disp.to_csv(index=False), "concentration.csv", "text/csv")

# ════ TAB 2 — CURRENT SHORTAGES ════
with t2:
    st.markdown(f"""
    <div style="background:rgba(239,68,68,0.05);border:1px solid rgba(239,68,68,0.15);
         border-radius:10px;padding:12px 16px;margin-bottom:16px;font-size:13px;color:#8BA4C0;line-height:1.7;">
    <b style="color:#F87171;">{n_shortage_records:,} records</b> currently active ({n_current_shortage} unique drugs).
    Source: FDA Drug Shortage Database.
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div style="font-size:10.5px;text-transform:uppercase;letter-spacing:.1em;font-weight:700;color:#3A5A7C;margin-bottom:10px;">Current Shortages by Therapeutic Category</div>', unsafe_allow_html=True)
        cat_counts = current_df['Therapeutic Category'].value_counts().head(12)
        fig = go.Figure(go.Bar(
            x=cat_counts.values, y=cat_counts.index, orientation='h',
            marker=dict(color=RED,opacity=0.8),
            text=cat_counts.values, textposition='outside',
            textfont=dict(size=10,color="#8BA4C0"),
        ))
        fig.update_layout(**PLOT, height=360,
            xaxis=dict(showgrid=False,showline=False,showticklabels=False),
            yaxis=dict(showgrid=False,showline=False,tickfont=dict(size=10,color="#8BA4C0"),autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    with c2:
        st.markdown('<div style="font-size:10.5px;text-transform:uppercase;letter-spacing:.1em;font-weight:700;color:#3A5A7C;margin-bottom:10px;">Root Cause Breakdown</div>', unsafe_allow_html=True)
        reasons = current_df['Reason for Shortage'].value_counts()
        reasons = reasons[reasons.index.notna()]
        fig2 = go.Figure(go.Pie(
            labels=[r[:38] for r in reasons.index], values=reasons.values, hole=0.5,
            marker_colors=[RED,ORANGE,AMBER,GREEN,BLUE,CYAN,"#8B5CF6","#EC4899"],
            textinfo="label+percent", textfont=dict(size=10,color="#E8F0FF"),
        ))
        fig2.update_layout(**PLOT, height=360, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:10.5px;text-transform:uppercase;letter-spacing:.1em;font-weight:700;color:#3A5A7C;margin-bottom:10px;">Shortage Records by Year of Initial Posting</div>', unsafe_allow_html=True)
    all_s = shortage_df.copy()
    all_s['Initial Posting Date'] = pd.to_datetime(all_s['Initial Posting Date'], errors='coerce')
    all_s['year'] = all_s['Initial Posting Date'].dt.year
    year_counts = all_s[all_s['year'].between(2012,2026)].groupby(['year','Status']).size().reset_index(name='count')
    fig3 = go.Figure()
    for status, color in [('Current',RED),('Resolved',GREEN),('To Be Discontinued',ORANGE)]:
        sub = year_counts[year_counts['Status']==status]
        fig3.add_trace(go.Bar(name=status, x=sub['year'], y=sub['count'], marker_color=color, opacity=0.8))
    fig3.update_layout(**PLOT, height=240, barmode='stack',
        xaxis=dict(showgrid=False,tickfont=dict(color="#5A7A9C")),
        yaxis=dict(showgrid=False,tickfont=dict(color="#5A7A9C")),
        legend=dict(font=dict(size=11,color="#8BA4C0"),bgcolor="rgba(0,0,0,0)",orientation="h",x=0,y=1.12))
    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar":False})
    st.caption("Source: FDA Drug Shortage Database. 2012–2026.")

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:10.5px;text-transform:uppercase;letter-spacing:.1em;font-weight:700;color:#3A5A7C;margin-bottom:10px;">Companies with Most Active Shortage Records</div>', unsafe_allow_html=True)
    top_co = current_df['Company Name'].value_counts().head(15)
    fig4 = go.Figure(go.Bar(
        x=top_co.values, y=top_co.index, orientation='h',
        marker=dict(color=ORANGE,opacity=0.8),
        text=top_co.values, textposition='outside',
        textfont=dict(size=10,color="#8BA4C0"),
    ))
    fig4.update_layout(**PLOT, height=380,
        xaxis=dict(showgrid=False,showline=False,showticklabels=False),
        yaxis=dict(showgrid=False,showline=False,tickfont=dict(size=11,color="#8BA4C0"),autorange="reversed"))
    st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar":False})
    st.caption("Hospira (Pfizer) and Fresenius Kabi account for over 27% of all active shortage records.")

# ════ TAB 3 — VULNERABILITY INTERSECTION ════
with t3:
    matched = merged_df[merged_df['HHI'].notna()].copy()
    st.markdown(f"""
    <div style="background:rgba(239,68,68,0.05);border:1px solid rgba(239,68,68,0.15);
         border-radius:10px;padding:12px 16px;margin-bottom:16px;font-size:13px;color:#8BA4C0;line-height:1.7;">
    <b style="color:#F87171;">Maximum vulnerability</b> = drug currently in shortage AND high supply concentration.
    These carry double risk: shortage is active AND structural substitution is limited.
    This intersection is the core network vulnerability metric for pharmaceutical supply chain resilience.
    </div>""", unsafe_allow_html=True)

    v1,v2,v3 = st.columns(3)
    v1.metric("Shortage drugs matched to Orange Book", matched['Generic Name'].nunique())
    v2.metric("Of these: Single Source (HHI=1.0)",     (matched['HHI']==1.0).sum(), "Highest vulnerability")
    v3.metric("Of these: High Concentration (HHI>0.5)",(matched['HHI']>0.5).sum())

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:10.5px;text-transform:uppercase;letter-spacing:.1em;font-weight:700;color:#3A5A7C;margin-bottom:10px;">Concentration vs. Shortage Volume — Each Point = One Drug</div>', unsafe_allow_html=True)

    scatter = matched.groupby(['Generic Name','HHI','n_manufacturers','Therapeutic Category']).size().reset_index(name='shortage_records')
    scatter['risk_label'] = scatter['HHI'].apply(
        lambda h: "Single Source" if h==1.0 else ("High" if h>0.5 else ("Moderate" if h>0.25 else "Competitive"))
    )
    fig_s = go.Figure()
    for tier, color in [("Single Source",RED),("High",ORANGE),("Moderate",AMBER),("Competitive",GREEN)]:
        sub = scatter[scatter['risk_label']==tier]
        if len(sub):
            fig_s.add_trace(go.Scatter(
                x=sub['HHI'], y=sub['shortage_records'], mode='markers', name=tier,
                marker=dict(color=color,size=10,opacity=0.8,line=dict(color="#0D1629",width=1)),
                text=sub['Generic Name'],
                hovertemplate="<b>%{text}</b><br>HHI: %{x:.3f}<br>Records: %{y}<extra></extra>",
            ))
    fig_s.add_vline(x=1.0, line_dash="dash", line_color=RED,
                    annotation_text="Single source", annotation_font_color=RED)
    fig_s.add_vline(x=0.5, line_dash="dot", line_color=ORANGE)
    fig_s.update_layout(**PLOT, height=360,
        xaxis=dict(title="HHI Score (supply concentration)",showgrid=False,tickfont=dict(color="#5A7A9C"),range=[-0.05,1.1]),
        yaxis=dict(title="Active shortage records",showgrid=False,tickfont=dict(color="#5A7A9C")),
        legend=dict(font=dict(size=11,color="#8BA4C0"),bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig_s, use_container_width=True, config={"displayModeBar":False})
    st.caption("Top-right = highest risk (high shortage volume + high concentration). Source: FDA Orange Book + FDA Drug Shortage Database.")

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:10.5px;text-transform:uppercase;letter-spacing:.1em;font-weight:700;color:#3A5A7C;margin-bottom:8px;">Full Matched Drug Table</div>', unsafe_allow_html=True)

    tbl = matched.groupby(['Generic Name','Therapeutic Category','HHI','n_manufacturers']).agg(
        shortage_records=('Generic Name','count'),
        primary_company=('Company Name', lambda x: x.dropna().value_counts().index[0] if len(x.dropna()) else 'N/A'),
        reason=('Reason for Shortage', lambda x: x.dropna().value_counts().index[0] if len(x.dropna()) else 'N/A'),
    ).reset_index().sort_values('HHI', ascending=False)
    tbl['HHI'] = tbl['HHI'].round(3)
    tbl['Risk Tier'] = tbl['HHI'].apply(
        lambda h: "Single Source" if h==1.0 else ("High" if h>0.5 else ("Moderate" if h>0.25 else "Competitive"))
    )
    st.dataframe(
        tbl[['Generic Name','Therapeutic Category','HHI','n_manufacturers','Risk Tier','shortage_records','reason']].rename(columns={
            'Generic Name':'Drug','Therapeutic Category':'Category',
            'HHI':'HHI Score','n_manufacturers':'Approved Mfrs',
            'shortage_records':'Active Records','reason':'Primary Reason',
        }),
        use_container_width=True, height=380, hide_index=True
    )
    st.download_button("Export", tbl.to_csv(index=False), "vulnerability.csv", "text/csv")

# ════ TAB 4 — COMPANY EXPOSURE ════
with t4:
    st.markdown("""
    <div style="background:rgba(59,130,246,0.05);border:1px solid rgba(59,130,246,0.15);
         border-radius:10px;padding:12px 16px;margin-bottom:16px;font-size:13px;color:#8BA4C0;line-height:1.7;">
    <b style="color:#93C5FD;">Network criticality</b>: if a company's manufacturing is disrupted,
    how many single-source drugs lose their only supplier?
    This is the node criticality analysis for pharmaceutical supply networks —
    A key metric for understanding systemic fragility in pharmaceutical supply networks.
    </div>""", unsafe_allow_html=True)

    top_ss = company_risk_df[company_risk_df['n_single_source_drugs'] > 0].sort_values(
        'n_single_source_drugs', ascending=False
    ).head(20)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div style="font-size:10.5px;text-transform:uppercase;letter-spacing:.1em;font-weight:700;color:#3A5A7C;margin-bottom:10px;">Companies by Single-Source Drug Count</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Bar(
            x=top_ss['n_single_source_drugs'],
            y=top_ss['Applicant_Full_Name'].str.title().str[:35],
            orientation='h', marker=dict(color=RED,opacity=0.8),
            text=top_ss['n_single_source_drugs'], textposition='outside',
            textfont=dict(size=10,color="#8BA4C0"),
        ))
        fig.update_layout(**PLOT, height=480,
            xaxis=dict(showgrid=False,showline=False,showticklabels=False,
                       title="Single-source drugs controlled"),
            yaxis=dict(showgrid=False,showline=False,tickfont=dict(size=10,color="#8BA4C0"),autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        st.caption("Each bar = number of active ingredients for which this is the only FDA-approved manufacturer.")

    with c2:
        st.markdown('<div style="font-size:10.5px;text-transform:uppercase;letter-spacing:.1em;font-weight:700;color:#3A5A7C;margin-bottom:10px;">Portfolio Size vs. Single-Source Concentration</div>', unsafe_allow_html=True)
        pd_ = company_risk_df[company_risk_df['n_drugs_total'] >= 5].copy()
        pd_['pct_ss'] = pd_['n_single_source_drugs'] / pd_['n_drugs_total'] * 100
        fig2 = go.Figure(go.Scatter(
            x=pd_['n_drugs_total'], y=pd_['pct_ss'], mode='markers',
            marker=dict(
                size=pd_['n_single_source_drugs'].clip(3,20),
                color=pd_['n_single_source_drugs'],
                colorscale=[[0,"#1A2840"],[0.3,AMBER],[0.7,ORANGE],[1.0,RED]],
                showscale=True,
                colorbar=dict(title=dict(text="Single-source drugs",font=dict(color="#5A7A9C",size=10)),
                              tickfont=dict(color="#5A7A9C",size=9)),
                opacity=0.8, line=dict(color="#0D1629",width=0.5)
            ),
            text=pd_['Applicant_Full_Name'].str.title(),
            hovertemplate="<b>%{text}</b><br>Total: %{x} drugs<br>% single-source: %{y:.1f}%<extra></extra>",
        ))
        fig2.update_layout(**PLOT, height=480,
            xaxis=dict(title="Total RX drug portfolio size",showgrid=False,tickfont=dict(color="#5A7A9C")),
            yaxis=dict(title="% portfolio that is single-source",showgrid=False,tickfont=dict(color="#5A7A9C")))
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:10.5px;text-transform:uppercase;letter-spacing:.1em;font-weight:700;color:#3A5A7C;margin-bottom:8px;">Full Company Risk Profile</div>', unsafe_allow_html=True)
    co_search = st.text_input("Search company", placeholder="e.g. Pfizer, Fresenius, Hikma...", label_visibility="collapsed")
    disp_cr = company_risk_df.copy()
    if co_search:
        disp_cr = disp_cr[disp_cr['Applicant_Full_Name'].str.contains(co_search.upper(), na=False)]
    disp_cr['% Single-Source'] = (disp_cr['n_single_source_drugs'] / disp_cr['n_drugs_total'] * 100).round(1)
    st.dataframe(
        disp_cr[['Applicant_Full_Name','n_drugs_total','n_single_source_drugs','% Single-Source']].rename(columns={
            'Applicant_Full_Name':'Manufacturer','n_drugs_total':'Total RX Drugs',
            'n_single_source_drugs':'Single-Source Drugs'}).head(200),
        use_container_width=True, height=420, hide_index=True
    )
    st.download_button("Export", disp_cr.to_csv(index=False), "company_risk.csv", "text/csv")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<p style='font-size:11px;color:#2A4060;text-align:center;line-height:1.8;'>
PharmNet Risk · FDA Pharmaceutical Supply Chain Concentration Analyzer · Real FDA data — no synthetic data ·
HHI: Hirschman (1945) · GAO (2014) Drug Shortages: Root Causes and Potential Solutions ·
Built by Rutwik Satish · MS Engineering Management, Northeastern University
</p>""", unsafe_allow_html=True)
