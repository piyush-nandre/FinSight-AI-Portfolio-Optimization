import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import plotting

from news_pipeline import fetch_portfolio_news
from embeddings_pipeline import generate_news_embeddings
from faiss_pipeline import (
    build_faiss_index,
    search_news,
)

from gemini_pipeline import (
    generate_portfolio_insight
)

from stress_testing import (
    run_stress_test
)

from recommendation_engine import (
    generate_recommendation
)


# ===========================
# PREMIUM UI STYLING & CUSTOM LIGHT THEME
# ===========================
st.set_page_config(
    page_title="FinSight AI",
    page_icon="📈",
    layout="wide",
)

# High-end Custom CSS Injection
st.markdown("""
    <style>
        /* Modern Soft Canvas Gradient Background */
        .stApp {
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%) !important;
        }
        
        /* Premium Card styling for Metrics */
        div[data-testid="stMetricContainer"] {
            background-color: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
            padding: 24px 28px !important;
            border-radius: 16px !important;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.03), 0 1px 4px rgba(15, 23, 42, 0.02) !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        div[data-testid="stMetricContainer"]:hover {
            box-shadow: 0 10px 20px rgba(15, 23, 42, 0.05) !important;
            transform: translateY(-2px);
        }
        
        /* Elegant typography tuning */
        h1, h2, h3 {
            color: #0f172a !important;
            font-weight: 700 !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
        }
        
        /* Modern Button Architecture */
        .stButton>button {
            border-radius: 12px !important;
            font-weight: 600 !important;
            padding: 0.6rem 2rem !important;
            transition: all 0.2s ease !important;
        }
        
        /* Soften info/success alert panels to match premium palette */
        div[data-testid="stNotification"] {
            border-radius: 12px !important;
            border: 1px solid rgba(0,0,0,0.05) !important;
        }
        
        /* Minimal Tab UI Clean-up */
        button[data-baseweb="tab"] {
            font-weight: 600 !important;
            font-size: 16px !important;
        }
    </style>
""", unsafe_allow_html=True)



# ===========================
# PAGE CONFIGURATION & LIGHT THEME
# ===========================
st.set_page_config(
    page_title="FinSight AI",
    page_icon="📈",
    layout="wide",
)

st.title("📈 FinSight AI")
st.subheader("Intelligent Indian Portfolio Optimization Platform")

# ===========================
# TOP CONTROLS (NO SIDEBAR)
# ===========================
st.markdown("### 🛠️ Portfolio Configuration")
controls_col1, controls_col2, controls_col3 = st.columns([1, 1, 2])

with controls_col1:
    investment_amount = st.number_input(
        "Investment Amount (₹)",
        min_value=1000,
        value=100000,
        step=1000,
    )

with controls_col2:
    risk_profile = st.selectbox(
        "Risk Profile",
        ["Conservative", "Moderate", "Aggressive"],
    )

with controls_col3:
    default_assets = [
        "RELIANCE.NS",
        "HDFCBANK.NS",
        "ICICIBANK.NS",
        "INFY.NS",
        "TCS.NS",
        "LT.NS",
        "NIFTYBEES.NS",
        "BANKBEES.NS",
        "GOLDBEES.NS",
    ]
    selected_assets = st.multiselect(
        "Select Stocks and ETFs",
        options=default_assets,
        default=default_assets,
    )

# Initialization & Trigger
if "portfolio_generated" not in st.session_state:
    st.session_state.portfolio_generated = False

btn_col1, _ = st.columns([1, 3])
with btn_col1:
    if st.button("Generate Portfolio", use_container_width=True, type="primary"):
        st.session_state.portfolio_generated = True

generate = st.session_state.portfolio_generated

# ===========================
# MAIN CORE LOGIC
# ===========================
if generate:
    if len(selected_assets) < 2:
        st.error("Please select at least 2 assets.")
    else:
        try:
            with st.spinner("Optimizing portfolio metrics..."):
                data = yf.download(
                    selected_assets,
                    start="2020-01-01",
                    auto_adjust=True,
                    progress=False,
                )["Close"]

                data = data.ffill().dropna()

                mu = mean_historical_return(data)
                S = CovarianceShrinkage(data).ledoit_wolf()

                ef = EfficientFrontier(mu, S)

                if risk_profile == "Conservative":
                    ef.min_volatility()
                elif risk_profile == "Moderate":
                    ef.max_sharpe()
                else:
                    ef.max_quadratic_utility()

                weights = ef.clean_weights()
                expected_return, volatility, sharpe = ef.portfolio_performance()

            st.divider()

            # ===========================
            # 1. SUMMARY CARDS
            # ===========================
            st.markdown("### 📊 Portfolio Summary")
            summary_col1, summary_col2, summary_col3 = st.columns(3)

            summary_col1.metric("Investment Amount", f"₹{investment_amount:,.0f}")
            summary_col2.metric("Risk Profile", risk_profile)
            summary_col3.metric("Assets Selected", len(selected_assets))
            
            st.info("💡 **Selected Assets:** " + " • ".join(selected_assets))
            st.divider()

            # ===========================
            # TABBED CONTAINER VIEW
            # ===========================
            tab1, tab2, tab3, tab4 = st.tabs(
                [
                    "📈 Portfolio",
                    "🎯 Recommendation",
                    "📉 Stress Test",
                    "📰 AI Insights & News"
                ]
            )

            # ---------------------------
            # TAB 1: PORTFOLIO
            # ---------------------------
            with tab1:
                st.markdown("### Portfolio Performance")
                perf_col1, perf_col2, perf_col3 = st.columns(3)
                perf_col1.metric("Expected Return", f"{expected_return:.2%}")
                perf_col2.metric("Volatility", f"{volatility:.2%}")
                perf_col3.metric("Sharpe Ratio", f"{sharpe:.2f}")

                allocation_df = pd.DataFrame(
                    {
                        "Asset": list(weights.keys()),
                        "Weight": list(weights.values()),
                    }
                )
                allocation_df = allocation_df[allocation_df["Weight"] > 0].copy()

                if allocation_df.empty:
                    st.warning("No valid allocation generated.")
                else:
                    allocation_df["Weight (%)"] = allocation_df["Weight"] * 100
                    allocation_df["Clean Asset"] = allocation_df["Asset"].str.replace(".NS", "", regex=False)
                    allocation_df["Amount (₹)"] = allocation_df["Weight"] * investment_amount

                    st.divider()
                    st.markdown("### Portfolio Composition")
                    
                    donut_col, table_col = st.columns([1, 1])
                    
                    with donut_col:
                        fig_donut, ax_donut = plt.subplots(figsize=(6, 4))
                        wedges, texts, autotexts = ax_donut.pie(
                            allocation_df["Weight (%)"],
                            labels=None,
                            autopct="%1.1f%%",
                            startangle=90,
                            pctdistance=0.75,
                            wedgeprops=dict(width=0.35),
                        )
                        ax_donut.axis("equal")
                        ax_donut.legend(
                            wedges,
                            allocation_df["Clean Asset"],
                            title="Assets",
                            loc="center left",
                            bbox_to_anchor=(0.9, 0.5),
                        )
                        st.pyplot(fig_donut)

                    with table_col:
                        st.markdown("**Optimal & Investment Allocation**")
                        st.dataframe(
                            allocation_df[["Clean Asset", "Weight (%)", "Amount (₹)"]].rename(
                                columns={"Clean Asset": "Asset"}
                            ),
                            use_container_width=True,
                            hide_index=True
                        )

                    st.divider()
                    st.markdown("### Efficient Frontier Visualization")
                    ef_plot = EfficientFrontier(mu, S)
                    fig_ef, ax_ef = plt.subplots(figsize=(10, 5))
                    
                    plotting.plot_efficient_frontier(ef_plot, ax=ax_ef, show_assets=True)
                    asset_volatility = data.pct_change().dropna().std() * (252 ** 0.5)

                    for asset in mu.index:
                        clean_name = asset.replace(".NS", "")
                        ax_ef.annotate(
                            clean_name,
                            (asset_volatility[asset], mu[asset]),
                            xytext=(5, 5),
                            textcoords="offset points",
                            fontsize=8,
                        )

                    portfolio_point = ax_ef.scatter(
                        volatility, expected_return, marker="*", s=300, color="red", label="Your Portfolio", zorder=10
                    )
                    frontier_handle = plt.Line2D([], [], color="C0", linewidth=2, label="Efficient Frontier")
                    assets_handle = plt.Line2D([], [], marker="o", linestyle="", color="C1", markersize=8, label="Assets")
                    
                    ax_ef.legend(handles=[frontier_handle, assets_handle, portfolio_point])
                    ax_ef.set_xlabel("Annual Volatility")
                    ax_ef.set_ylabel("Expected Annual Return")
                    st.pyplot(fig_ef)

            # ---------------------------
            # TAB 2: RECOMMENDATION
            # ---------------------------
            with tab2:
                temp_stress_results = run_stress_test(weights, "RBI Rate Hike")
                
                recommendation = generate_recommendation(
                    expected_return=expected_return,
                    volatility=volatility,
                    sharpe=sharpe,
                    stress_impact=temp_stress_results["impact"],
                    risk_profile=risk_profile
                )

                recommendation_text = {
                    "STRONG BUY": "Portfolio fundamentals and risk-adjusted returns are very strong.",
                    "BUY": "Portfolio appears healthy with favorable risk-return characteristics.",
                    "HOLD": "Portfolio is balanced but has limited upside based on current metrics.",
                    "REDUCE": "Portfolio shows elevated risk relative to expected returns."
                }

                st.markdown("### Portfolio Diagnostics")
                rec_col1, rec_col2 = st.columns([1, 1])

                with rec_col1:
                    # PERFECT ALIGNMENT: Uses domain to center both the graphic ring and value readout box
                    fig_gauge = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=recommendation["score"],
                            title={"text": "Portfolio Health Score", "font": {"size": 18}},
                            number={"font": {"size": 52}},
                            gauge={
                                "axis": {"range": [0, 100], "tickwidth": 1},
                                "bar": {"color": "#1f77b4"},
                            },
                            domain={"x": [0, 1], "y": [0, 1]}
                        )
                    )
                    fig_gauge.update_layout(
                        height=280, 
                        margin=dict(l=50, r=50, t=50, b=20)
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)

                with rec_col2:
                    st.metric("System Recommendation", recommendation["recommendation"])
                    st.info(recommendation_text.get(recommendation["recommendation"], ""))

                st.divider()
                st.markdown("### Analysis Breakdown")
                breakdown_col1, breakdown_col2 = st.columns(2)

                with breakdown_col1:
                    st.markdown("#### 💪 Strengths")
                    if recommendation["strengths"]:
                        for item in recommendation["strengths"]:
                            st.success(item)
                    else:
                        st.info("No major strengths identified.")

                with breakdown_col2:
                    st.markdown("#### ⚠️ Risks")
                    if recommendation["risks"]:
                        for item in recommendation["risks"]:
                            st.warning(item)
                    else:
                        st.success("No significant risks detected.")

            # ---------------------------
            # TAB 3: STRESS TESTING
            # ---------------------------
            with tab3:
                st.markdown("### Macroeconomic Stress Testing")
                
                scenario = st.selectbox(
                    "Select Stress Scenario",
                    [
                        "RBI Rate Hike", "IT Slowdown", "Market Correction", 
                        "Bull Market", "Recession", "Oil Price Shock", 
                        "Banking Crisis", "AI Boom", "China Slowdown"
                    ]
                )

                stress_results = run_stress_test(weights, scenario)
                impact = stress_results["impact"]

                if abs(impact) < 0.02:
                    risk_label = "Low Impact"
                elif abs(impact) < 0.05:
                    risk_label = "Moderate Impact"
                else:
                    risk_label = "High Impact"

                stress_col1, stress_col2 = st.columns(2)
                stress_col1.metric("Projected Portfolio Impact", f"{impact:.2%}")
                stress_col2.metric("Risk Assessment Profile", risk_label)

                scenario_explanations = {
                    "RBI Rate Hike": "Higher interest rates may pressure banking and rate-sensitive sectors.",
                    "IT Slowdown": "Technology companies may face lower earnings growth.",
                    "Market Correction": "Broad equity decline with defensive assets potentially outperforming.",
                    "Bull Market": "Strong market environment benefiting most equity holdings.",
                    "Recession": "Systemic down-cycle impacting overall market activity negatively.",
                    "Oil Price Shock": "Input costs surges might hit margins of manufacturing sectors.",
                    "Banking Crisis": "Liquidity bottlenecks hitting foundational financial infrastructure.",
                    "AI Boom": "Accelerated capital flow expanding technological capability valuations.",
                    "China Slowdown": "Global trade deceleration dampening external demand models."
                }
                st.info(scenario_explanations.get(scenario, "Custom economic scenario analysis model applied."))

                if "affected_assets" in stress_results and stress_results["affected_assets"]:
                    st.divider()
                    st.markdown("### Asset Impact Insights")
                    
                    asset_col1, asset_col2 = st.columns(2)
                    
                    # FORCED EXTRACTION AND SEPARATION LOGIC
                    sorted_losers = stress_results["top_losers"]
                    sorted_winners = stress_results["top_beneficiaries"]

                    with asset_col1:
                        st.markdown("#### 📉 Top Losers")

                        if sorted_losers:
                            for asset, imp in sorted_losers:
                                color = "red" if imp < 0 else "black"
                                st.markdown(
                                    f"**{asset.replace('.NS', '')}**: "
                                    f"<span style='color:{color}'>{imp:.2%}</span>",
                                    unsafe_allow_html=True
                                )
                        else:
                            st.info("No negatively impacted assets.")

                    with asset_col2:
                        st.markdown("#### 📈 Top Beneficiaries")

                        if sorted_winners:
                            for asset, imp in sorted_winners:
                                color = "green" if imp > 0 else "black"
                                st.markdown(
                                    f"**{asset.replace('.NS', '')}**: "
                                    f"<span style='color:{color}'>{imp:.2%}</span>",
                                    unsafe_allow_html=True
                                )
                        else:
                            st.success("No assets benefit from this scenario.")
            # ---------------------------
            # TAB 4: AI INSIGHTS & NEWS
            # ---------------------------
            with tab4:
                st.markdown("### Personalized Market News")
                try:
                    news_df = fetch_portfolio_news(selected_assets)
                    embedding_results = generate_news_embeddings(news_df)
                    faiss_index = build_faiss_index(embedding_results["embeddings"])

                    st.session_state.faiss_index = faiss_index
                    st.session_state.embedding_results = embedding_results

                    if news_df.empty:
                        st.info("No market news available at the moment.")
                    else:
                        st.markdown("#### 🔍 Contextual Search Engine")
                        with st.form("news_search_form"):
                            user_query = st.text_input("Ask about market events, specific sectors, or asset factors:")
                            search_clicked = st.form_submit_button("Query Engine")

                        if search_clicked:
                            retrieved_chunks = search_news(
                                user_query,
                                st.session_state.faiss_index,
                                st.session_state.embedding_results["chunks"],
                            )

                            if retrieved_chunks:
                                st.success(f"Retrieved {len(retrieved_chunks)} relevant contextual vectors.")
                                ai_response = generate_portfolio_insight(
                                    question=user_query,
                                    retrieved_chunks=retrieved_chunks,
                                    selected_assets=selected_assets,
                                    risk_profile=risk_profile
                                )
                                st.markdown("##### 🤖 AI Gen-Insight Response")
                                st.write(ai_response)
                            else:
                                st.warning("No highly relevant vectors matched the search text.")
                        
                        st.divider()
                        
                        st.markdown("#### Pipeline State Metrics")
                        status_col1, status_col2, status_col3 = st.columns(3)
                        status_col1.metric("Processed Articles", len(embedding_results["documents"]))
                        status_col2.metric("Extracted Context Chunks", len(embedding_results["chunks"]))
                        status_col3.metric("Vector Dimensionality", embedding_results["embedding_dimension"])
                        
                        st.divider()
                        st.markdown("#### Latest Bulletins")
                        for _, row in news_df.head(6).iterrows():
                            st.markdown(f"**{row['Headline']}**")
                            st.caption(f"{row['Source']} • {row['Published']} — [Source Link]({row['Link']})")
                            st.markdown("")

                except Exception as e:
                    st.warning(f"Unable to safely pull live vector data streams: {e}")

        except Exception as e:
            st.error(f"Execution dynamic error encountered: {str(e)}")
else:
    st.info("Configure your target financial metrics using the input controllers above and click 'Generate Portfolio'.")