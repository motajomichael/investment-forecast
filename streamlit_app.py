import streamlit as st
import pandas as pd

# -----------------------
# Core simulation logic
# -----------------------

def simulate_joint_isa(
    start_year: int,
    end_year: int,
    you_start_balance: float,
    wife_start_balance: float,
    you_initial_contrib: float,
    wife_initial_contrib: float,
    annual_raise_pct: float,
    annual_return_pct: float,
    you_cap: float,
    wife_cap: float,
    redirect_to_wife: bool,
):
    """Simulate year-by-year joint ISA balances and contributions (person-level)."""

    years = list(range(start_year, end_year + 1))
    n = len(years)

    raise_factor = annual_raise_pct / 100.0
    r = annual_return_pct / 100.0

    # Annual contributions per person
    you_contrib = [0.0] * n
    wife_contrib = [0.0] * n

    # Starting year contributions
    you_contrib[0] = min(you_initial_contrib, you_cap)
    wife_contrib[0] = min(wife_initial_contrib, wife_cap)

    # Future years: raise + cap + optional redirect
    for i in range(1, n):
        # Husband
        you_desired = you_contrib[i - 1] * (1 + raise_factor)
        you_contrib[i] = min(you_desired, you_cap)

        # Wife (base)
        wife_base_desired = wife_contrib[i - 1] * (1 + raise_factor)

        # Husband "ideal" next year if no cap
        you_desired_next = you_contrib[i - 1] * (1 + raise_factor)
        you_excess = 0.0
        if redirect_to_wife:
            you_excess = max(you_desired_next - you_cap, 0.0)

        raw_wife = wife_base_desired + you_excess

        if wife_contrib[i - 1] >= wife_cap:
            wife_contrib[i] = wife_cap
        else:
            wife_contrib[i] = min(raw_wife, wife_cap)

    # Balances
    you_start = [0.0] * n
    you_end = [0.0] * n
    wife_start = [0.0] * n
    wife_end = [0.0] * n

    you_start[0] = you_start_balance
    wife_start[0] = wife_start_balance

    for i in range(n):
        if i > 0:
            you_start[i] = you_end[i - 1]
            wife_start[i] = wife_end[i - 1]

        you_end[i] = (you_start[i] + you_contrib[i]) * (1 + r)
        wife_end[i] = (wife_start[i] + wife_contrib[i]) * (1 + r)

    rows = []
    cum_you = 0.0
    cum_wife = 0.0

    for i, year in enumerate(years):
        yc = you_contrib[i]
        wc = wife_contrib[i]
        cum_you += yc
        cum_wife += wc
        total_contrib = yc + wc

        combined_start = you_start[i] + wife_start[i]
        combined_end = you_end[i] + wife_end[i]
        cum_total = cum_you + cum_wife
        cum_growth = combined_end - cum_total

        rows.append(
            {
                "Year": year,
                "You_Annual_Contrib": yc,
                "Wife_Annual_Contrib": wc,
                "Combined_Annual_Contrib": total_contrib,
                "Monthly_Funding_Total": total_contrib / 12.0,
                "You_Start_Balance": you_start[i],
                "You_End_Balance": you_end[i],
                "Wife_Start_Balance": wife_start[i],
                "Wife_End_Balance": wife_end[i],
                "Combined_End_Balance": combined_end,
                "You_Cum_Deposits": cum_you,
                "Wife_Cum_Deposits": cum_wife,
                "Combined_Cum_Deposits": cum_total,
                "Combined_Cum_Growth": cum_growth,
            }
        )

    return pd.DataFrame(rows)


def build_schedule_df(df, you_ports, wife_ports, weeks_per_year, biweekly_per_year):
    """
    Given person-level df and portfolio configs, return a schedule of per-period contributions
    for each portfolio by year.
    """
    freq_to_periods = {
        "Weekly": weeks_per_year,
        "Biweekly": biweekly_per_year,
        "Monthly": 12,
    }

    # Normalise weights for each person
    total_you_w = sum(max(p["weight"], 0.0) for p in you_ports)
    total_wife_w = sum(max(p["weight"], 0.0) for p in wife_ports)

    you_norm = []
    for p in you_ports:
        if total_you_w > 0:
            you_norm.append({**p, "norm_weight": max(p["weight"], 0.0) / total_you_w})
        else:
            you_norm.append({**p, "norm_weight": 0.0})

    wife_norm = []
    for p in wife_ports:
        if total_wife_w > 0:
            wife_norm.append({**p, "norm_weight": max(p["weight"], 0.0) / total_wife_w})
        else:
            wife_norm.append({**p, "norm_weight": 0.0})

    rows = []
    for _, row in df.iterrows():
        year = int(row["Year"])
        r = {"Year": year}

        # Husband
        you_annual = row["You_Annual_Contrib"]
        for p in you_norm:
            label = p["label"]
            freq = p["freq"]
            w = p["norm_weight"]
            periods = freq_to_periods.get(freq, 0)
            annual_alloc = you_annual * w
            per_period = annual_alloc / periods if periods > 0 else 0.0
            col_name = f"H {label} ({freq})"
            r[col_name] = per_period

        # Wife
        wife_annual = row["Wife_Annual_Contrib"]
        for p in wife_norm:
            label = p["label"]
            freq = p["freq"]
            w = p["norm_weight"]
            periods = freq_to_periods.get(freq, 0)
            annual_alloc = wife_annual * w
            per_period = annual_alloc / periods if periods > 0 else 0.0
            col_name = f"W {label} ({freq})"
            r[col_name] = per_period

        rows.append(r)

    return pd.DataFrame(rows)


# -----------------------
# Streamlit UI
# -----------------------

st.set_page_config(page_title="Joint ISA Planner (Dynamic + Scenarios)", layout="wide")

st.title("Joint ISA Planner – Dynamic Portfolios & Scenarios")

st.markdown(
    """
This app models **both your ISAs together** with:

- Start balances, initial annual contributions, annual raises  
- ISA caps for each of you, plus optional **redirect** of your surplus to your wife's ISA  
- Up to **5 portfolios each**, with custom **% allocation** and **frequency** (weekly / biweekly / monthly)  
- **Base / Dip / Rise** scenarios for annual return  

It calculates person-level contributions & growth, then splits them into per-portfolio
per-period schedules for each year.
"""
)

with st.sidebar:
    st.header("Core Inputs")

    start_year = st.number_input("Start Year", value=2026, step=1)
    end_year = st.number_input("Target End Year", value=2042, step=1, min_value=start_year)

    st.subheader("Start Balances (£)")
    you_start_balance = st.number_input("Start Balance (Husband)", value=2513.0, step=100.0)
    wife_start_balance = st.number_input("Start Balance (Wife)", value=300.0, step=50.0)

    st.subheader("Initial Annual Contributions (£)")
    st.caption("First-year annual contributions in the Start Year.")
    you_initial_contrib = st.number_input("Initial Annual Contrib – Husband", value=5096.0, step=100.0)
    wife_initial_contrib = st.number_input("Initial Annual Contrib – Wife", value=1200.0, step=100.0)

    annual_raise_pct = st.number_input("Annual Raise % (Before Cap)", value=25.0, step=1.0, min_value=0.0, max_value=100.0)

    st.subheader("Return assumptions (%)")
    base_return_pct = st.number_input("Base Annual Return %", value=11.8, step=0.1)
    dip_delta = st.number_input("Dip scenario: subtract (%)", value=3.0, step=0.5)
    rise_delta = st.number_input("Rise scenario: add (%)", value=3.0, step=0.5)

    st.subheader("ISA Caps (£/yr)")
    you_cap = st.number_input("Husband ISA Cap", value=20000.0, step=1000.0)
    wife_cap = st.number_input("Wife ISA Cap", value=20000.0, step=1000.0)

    redirect_to_wife = st.checkbox("Redirect surplus to wife's ISA once you hit your cap?", value=True)

    st.subheader("Frequencies (per year)")
    weeks_per_year = st.number_input("Weeks per Year", value=52, step=1, min_value=1)
    biweekly_per_year = st.number_input("Biweekly Periods per Year", value=26, step=1, min_value=1)

st.markdown("### Husband Portfolios (up to 5)")
you_ports = []
cols_you = st.columns(5)
default_you_weights = [40.0, 30.0, 20.0, 5.0, 5.0]
default_you_freqs = ["Weekly", "Weekly", "Biweekly", "Monthly", "Monthly"]

for i in range(5):
    with cols_you[i]:
        label = st.text_input(f"H P{i+1} label", value=f"P{i+1}", key=f"you_label_{i}")
        weight = st.number_input(
            f"{label} % of his ISA",
            value=default_you_weights[i],
            step=5.0,
            key=f"you_weight_{i}",
        )
        freq = st.selectbox(
            f"{label} frequency",
            options=["Weekly", "Biweekly", "Monthly"],
            index=["Weekly", "Biweekly", "Monthly"].index(default_you_freqs[i]),
            key=f"you_freq_{i}",
        )
        you_ports.append({"label": label, "weight": weight, "freq": freq})

st.markdown("### Wife Portfolios (up to 5)")
wife_ports = []
cols_w = st.columns(5)
default_wife_weights = [100.0, 0.0, 0.0, 0.0, 0.0]
default_wife_freqs = ["Monthly", "Monthly", "Monthly", "Monthly", "Monthly"]

for i in range(5):
    with cols_w[i]:
        label = st.text_input(f"W P{i+1} label", value=f"W{i+1}", key=f"wife_label_{i}")
        weight = st.number_input(
            f"{label} % of her ISA",
            value=default_wife_weights[i],
            step=5.0,
            key=f"wife_weight_{i}",
        )
        freq = st.selectbox(
            f"{label} frequency",
            options=["Weekly", "Biweekly", "Monthly"],
            index=["Weekly", "Biweekly", "Monthly"].index(default_wife_freqs[i]),
            key=f"wife_freq_{i}",
        )
        wife_ports.append({"label": label, "weight": weight, "freq": freq})

run_btn = st.button("Run projection")

if run_btn:
    scenarios = {
        "Base": base_return_pct,
        "Dip": base_return_pct - dip_delta,
        "Rise": base_return_pct + rise_delta,
    }

    all_proj = []
    all_sched = []

    tabs = st.tabs(list(scenarios.keys()))

    for tab, (scenario_name, r_pct) in zip(tabs, scenarios.items()):
        with tab:
            st.subheader(f"{scenario_name} Scenario (Return = {r_pct:.1f}%)")

            df = simulate_joint_isa(
                start_year=start_year,
                end_year=end_year,
                you_start_balance=you_start_balance,
                wife_start_balance=wife_start_balance,
                you_initial_contrib=you_initial_contrib,
                wife_initial_contrib=wife_initial_contrib,
                annual_raise_pct=annual_raise_pct,
                annual_return_pct=r_pct,
                you_cap=you_cap,
                wife_cap=wife_cap,
                redirect_to_wife=redirect_to_wife,
            )
            df["Scenario"] = scenario_name

            st.markdown("**Year-by-year ISA projection**")
            main_cols = [
                "Year",
                "You_Annual_Contrib",
                "Wife_Annual_Contrib",
                "Combined_Annual_Contrib",
                "Monthly_Funding_Total",
                "You_Start_Balance",
                "You_End_Balance",
                "Wife_Start_Balance",
                "Wife_End_Balance",
                "Combined_End_Balance",
            ]

            st.dataframe(
                df[main_cols].style.format(
                    {
                        "You_Annual_Contrib": "£{:,.0f}",
                        "Wife_Annual_Contrib": "£{:,.0f}",
                        "Combined_Annual_Contrib": "£{:,.0f}",
                        "Monthly_Funding_Total": "£{:,.0f}",
                        "You_Start_Balance": "£{:,.0f}",
                        "You_End_Balance": "£{:,.0f}",
                        "Wife_Start_Balance": "£{:,.0f}",
                        "Wife_End_Balance": "£{:,.0f}",
                        "Combined_End_Balance": "£{:,.0f}",
                    }
                ),
                use_container_width=True,
            )

            st.markdown("**Per-portfolio per-period contributions by year**")
            sched_df = build_schedule_df(
                df,
                you_ports=you_ports,
                wife_ports=wife_ports,
                weeks_per_year=weeks_per_year,
                biweekly_per_year=biweekly_per_year,
            )
            sched_df["Scenario"] = scenario_name

            st.dataframe(
                sched_df.style.format(
                    {c: "£{:,.0f}" for c in sched_df.columns if c not in ["Year", "Scenario"]}
                ),
                use_container_width=True,
            )

            st.markdown("**Combined ISA balance over time**")
            chart_df = df[["Year", "Combined_End_Balance"]].set_index("Year")
            st.line_chart(chart_df)

            all_proj.append(df)
            all_sched.append(sched_df)

    # Export combined
    st.subheader("Export all scenarios")
    proj_concat = pd.concat(all_proj, ignore_index=True)
    sched_concat = pd.concat(all_sched, ignore_index=True)
    merged = proj_concat.merge(sched_concat, on=["Scenario", "Year"], how="left")

    csv = merged.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV (all scenarios + schedules)",
        data=csv,
        file_name="joint_isa_scenarios_dynamic_portfolios.csv",
        mime="text/csv",
    )
else:
    st.info("Set your inputs, portfolio weights and frequencies, then click **Run projection**.")
