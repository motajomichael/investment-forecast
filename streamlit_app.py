import streamlit as st
import pandas as pd

# -----------------------
# Core helpers
# -----------------------

def compute_contributions(
    start_year: int,
    end_year: int,
    you_initial_contrib: float,
    wife_initial_contrib: float,
    annual_raise_pct: float,
    you_cap: float,
    wife_cap: float,
    redirect_surplus: bool,
):
    """
    Compute year-by-year annual contributions for husband & wife, with caps and
    optional symmetric redirect:
      - If either spouse's desired contribution exceeds their cap,
        their surplus can be redirected to the other spouse (if the other is below cap).
    """
    years = list(range(start_year, end_year + 1))
    n = len(years)
    raise_factor = annual_raise_pct / 100.0

    you_contrib = [0.0] * n
    wife_contrib = [0.0] * n

    # Year 0 (start year) - apply caps, no redirect for simplicity
    you_contrib[0] = min(you_initial_contrib, you_cap)
    wife_contrib[0] = min(wife_initial_contrib, wife_cap)

    for i in range(1, n):
        # Desired contributions based on previous year (before caps/redirect)
        you_prev = you_contrib[i - 1]
        wife_prev = wife_contrib[i - 1]

        you_desired = you_prev * (1 + raise_factor)
        wife_desired = wife_prev * (1 + raise_factor)

        # Apply caps
        you_capped = min(you_desired, you_cap)
        wife_capped = min(wife_desired, wife_cap)

        if redirect_surplus:
            # Compute surplus for each
            you_excess = max(you_desired - you_cap, 0.0)
            wife_excess = max(wife_desired - wife_cap, 0.0)

            # Redirect husband's surplus to wife if she has room
            wife_room = max(wife_cap - wife_capped, 0.0)
            if you_excess > 0 and wife_room > 0:
                move = min(you_excess, wife_room)
                wife_capped += move

            # Redirect wife's surplus to husband if he has room
            you_room = max(you_cap - you_capped, 0.0)
            if wife_excess > 0 and you_room > 0:
                move = min(wife_excess, you_room)
                you_capped += move

        you_contrib[i] = you_capped
        wife_contrib[i] = wife_capped

    df = pd.DataFrame(
        {
            "Year": years,
            "You_Annual_Contrib": you_contrib,
            "Wife_Annual_Contrib": wife_contrib,
        }
    )
    return df


def simulate_balances_by_portfolios(
    contrib_df: pd.DataFrame,
    start_year: int,
    you_start_balance: float,
    wife_start_balance: float,
    you_ports: list,
    wife_ports: list,
    scenario_delta: float,
):
    """
    Given person-level contributions and per-portfolio base returns,
    simulate balances for one scenario (Base / Dip / Rise).
    scenario_delta is added to each portfolio's base ROR.
    """
    years = contrib_df["Year"].tolist()
    n = len(years)

    # Normalise weights
    total_you_w = sum(max(p["weight"], 0.0) for p in you_ports)
    total_wife_w = sum(max(p["weight"], 0.0) for p in wife_ports)

    you_norm = []
    for p in you_ports:
        w = max(p["weight"], 0.0)
        norm_w = w / total_you_w if total_you_w > 0 else 0.0
        you_norm.append({**p, "norm_weight": norm_w})

    wife_norm = []
    for p in wife_ports:
        w = max(p["weight"], 0.0)
        norm_w = w / total_wife_w if total_wife_w > 0 else 0.0
        wife_norm.append({**p, "norm_weight": norm_w})

    # Initial portfolio-level balances (split starting balances by weight)
    you_balances = [you_start_balance * p["norm_weight"] for p in you_norm]
    wife_balances = [wife_start_balance * p["norm_weight"] for p in wife_norm]

    rows = []
    cum_you = 0.0
    cum_wife = 0.0

    for i, year in enumerate(years):
        yc = contrib_df.loc[i, "You_Annual_Contrib"]
        wc = contrib_df.loc[i, "Wife_Annual_Contrib"]

        you_total_start = sum(you_balances)
        wife_total_start = sum(wife_balances)

        cum_you += yc
        cum_wife += wc

        # Update husband portfolios
        for idx, p in enumerate(you_norm):
            alloc = yc * p["norm_weight"]
            r = p["ror"] + scenario_delta  # annual ROR
            you_balances[idx] = (you_balances[idx] + alloc) * (1 + r / 100.0)

        # Update wife portfolios
        for idx, p in enumerate(wife_norm):
            alloc = wc * p["norm_weight"]
            r = p["ror"] + scenario_delta
            wife_balances[idx] = (wife_balances[idx] + alloc) * (1 + r / 100.0)

        you_total_end = sum(you_balances)
        wife_total_end = sum(wife_balances)
        combined_end = you_total_end + wife_total_end

        cum_total = cum_you + cum_wife
        cum_growth = combined_end - cum_total

        rows.append(
            {
                "Year": year,
                "You_Annual_Contrib": yc,
                "Wife_Annual_Contrib": wc,
                "Combined_Annual_Contrib": yc + wc,
                "Monthly_Funding_Total": (yc + wc) / 12.0,
                "You_Start_Balance": you_total_start,
                "You_End_Balance": you_total_end,
                "Wife_Start_Balance": wife_total_start,
                "Wife_End_Balance": wife_total_end,
                "Combined_End_Balance": combined_end,
                "You_Cum_Deposits": cum_you,
                "Wife_Cum_Deposits": cum_wife,
                "Combined_Cum_Deposits": cum_total,
                "Combined_Cum_Growth": cum_growth,
            }
        )

    return pd.DataFrame(rows)


def build_schedule_df(contrib_df, you_ports, wife_ports, weeks_per_year, biweekly_per_year):
    """Build per-period contribution schedule per portfolio, per year."""
    freq_to_periods = {
        "Weekly": weeks_per_year,
            "Biweekly": biweekly_per_year,
            "Monthly": 12,
    }

    total_you_w = sum(max(p["weight"], 0.0) for p in you_ports)
    total_wife_w = sum(max(p["weight"], 0.0) for p in wife_ports)

    you_norm = []
    for p in you_ports:
        w = max(p["weight"], 0.0)
        norm_w = w / total_you_w if total_you_w > 0 else 0.0
        you_norm.append({**p, "norm_weight": norm_w})

    wife_norm = []
    for p in wife_ports:
        w = max(p["weight"], 0.0)
        norm_w = w / total_wife_w if total_wife_w > 0 else 0.0
        wife_norm.append({**p, "norm_weight": norm_w})

    rows = []
    for _, row in contrib_df.iterrows():
        year = int(row["Year"])
        yc = row["You_Annual_Contrib"]
        wc = row["Wife_Annual_Contrib"]
        r = {"Year": year}

        # husband
        for p in you_norm:
            label = p["label"]
            freq = p["freq"]
            w = p["norm_weight"]
            periods = freq_to_periods.get(freq, 0)
            annual_alloc = yc * w
            per_period = annual_alloc / periods if periods > 0 else 0.0
            r[f"H {label} ({freq})"] = per_period

        # wife
        for p in wife_norm:
            label = p["label"]
            freq = p["freq"]
            w = p["norm_weight"]
            periods = freq_to_periods.get(freq, 0)
            annual_alloc = wc * w
            per_period = annual_alloc / periods if periods > 0 else 0.0
            r[f"W {label} ({freq})"] = per_period

        rows.append(r)

    return pd.DataFrame(rows)

# -----------------------
# Streamlit UI
# -----------------------

st.set_page_config(page_title="Joint ISA Planner – Scenarios & Dynamic Portfolios", layout="wide")

st.title("Joint ISA Planner – Dynamic Portfolios & Base / Dip / Rise Scenarios")

st.markdown(
    """
This app models **both your ISAs** with:

- Start balances & initial annual contributions  
- Annual raises and ISA caps, with optional **symmetric redirect** of surplus  
- Up to **5 portfolios each**, with custom % allocation, frequency, and **base annual return %**  
- Three scenarios: **Base**, **Dip**, and **Rise**  
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

    annual_raise_pct = st.number_input(
        "Annual Raise % (Before Cap)", value=25.0, step=1.0, min_value=0.0, max_value=100.0
    )

    st.subheader("Scenario Return Deltas (%)")
    st.caption("Each portfolio has a base ROR; Dip/Rise apply these deltas to all portfolios.")
    dip_delta = st.number_input("Dip scenario: subtract (%)", value=3.0, step=0.5)
    rise_delta = st.number_input("Rise scenario: add (%)", value=3.0, step=0.5)

    st.subheader("ISA Caps (£/yr)")
    you_cap = st.number_input("Husband ISA Cap", value=20000.0, step=1000.0)
    wife_cap = st.number_input("Wife ISA Cap", value=20000.0, step=1000.0)

    redirect_surplus = st.checkbox(
        "Redirect surplus contributions to the other spouse once cap is hit?",
        value=True,
    )

    st.subheader("Goal")
    target_goal = st.number_input("Target portfolio value (£)", value=1_000_000.0, step=50_000.0, min_value=0.0)

    st.subheader("Frequencies (per year)")
    weeks_per_year = st.number_input("Weeks per Year", value=52, step=1, min_value=1)
    biweekly_per_year = st.number_input("Biweekly Periods per Year", value=26, step=1, min_value=1)

st.markdown("### Husband Portfolios (up to 5)")
you_ports = []
cols_you = st.columns(5)
default_you_weights = [40.0, 30.0, 20.0, 5.0, 5.0]
default_you_freqs = ["Weekly", "Weekly", "Biweekly", "Monthly", "Monthly"]
default_you_rors = [11.8, 13.0, 9.0, 8.0, 7.0]

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
        ror = st.number_input(
            f"{label} base ROR (%)",
            value=default_you_rors[i],
            step=0.5,
            key=f"you_ror_{i}",
        )
        you_ports.append({"label": label, "weight": weight, "freq": freq, "ror": ror})

st.markdown("### Wife Portfolios (up to 5)")
wife_ports = []
cols_w = st.columns(5)
default_wife_weights = [100.0, 0.0, 0.0, 0.0, 0.0]
default_wife_freqs = ["Monthly", "Monthly", "Monthly", "Monthly", "Monthly"]
default_wife_rors = [11.8, 11.8, 11.8, 11.8, 11.8]

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
        ror = st.number_input(
            f"{label} base ROR (%)",
            value=default_wife_rors[i],
            step=0.5,
            key=f"wife_ror_{i}",
        )
        wife_ports.append({"label": label, "weight": weight, "freq": freq, "ror": ror})

run_btn = st.button("Run projection")

if run_btn:
    # 1) Compute contributions once (same for all scenarios)
    contrib_df = compute_contributions(
        start_year=start_year,
        end_year=end_year,
        you_initial_contrib=you_initial_contrib,
        wife_initial_contrib=wife_initial_contrib,
        annual_raise_pct=annual_raise_pct,
        you_cap=you_cap,
        wife_cap=wife_cap,
        redirect_surplus=redirect_surplus,
    )

    scenarios = {
        "Base": 0.0,
        "Dip": -dip_delta,
        "Rise": rise_delta,
    }

    all_proj = []
    all_sched = []

    tabs = st.tabs(list(scenarios.keys()))

    for tab, (scenario_name, delta) in zip(tabs, scenarios.items()):
        with tab:
            st.subheader(f"{scenario_name} Scenario (portfolio RORs {delta:+.1f} pts)")

            proj_df = simulate_balances_by_portfolios(
                contrib_df=contrib_df,
                start_year=start_year,
                you_start_balance=you_start_balance,
                wife_start_balance=wife_start_balance,
                you_ports=you_ports,
                wife_ports=wife_ports,
                scenario_delta=delta,
            )
            proj_df["Scenario"] = scenario_name

            # ---- QUICK SUMMARY BOX ----
            final_row = proj_df.iloc[-1]
            final_value = final_row["Combined_End_Balance"]
            total_deposits = final_row["Combined_Cum_Deposits"]
            total_growth = final_row["Combined_Cum_Growth"]
            years = end_year - start_year + 1

            invested = you_start_balance + wife_start_balance + total_deposits
            if invested > 0 and final_value > 0:
                cagr = ((final_value / invested) ** (1 / years) - 1) * 100
            else:
                cagr = 0.0

            if target_goal > 0:
                goal_hits = proj_df.loc[proj_df["Combined_End_Balance"] >= target_goal, "Year"]
                goal_year = int(goal_hits.iloc[0]) if not goal_hits.empty else "Not reached"
            else:
                goal_year = "N/A"

            monthly_avg = proj_df["Monthly_Funding_Total"].mean()

            st.markdown("### 📊 Quick Scenario Summary")
            st.info(
                f"""
**Final Combined Value:** £{final_value:,.0f}  
**Total Deposits:** £{total_deposits:,.0f}  
**Total Growth:** £{total_growth:,.0f}  
**Effective CAGR (approx):** {cagr:.2f}%  
**Goal (£{target_goal:,.0f}) Reached In Year:** {goal_year}  
**Average Monthly Funding Needed:** £{monthly_avg:,.0f}  
                """
            )
            st.markdown("---")

            # Main table
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
                proj_df[main_cols].style.format(
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

            # Per-portfolio schedule
            st.markdown("**Per-portfolio per-period contributions by year**")
            sched_df = build_schedule_df(
                contrib_df,
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

            # Chart
            st.markdown("**Combined ISA balance over time**")
            chart_df = proj_df[["Year", "Combined_End_Balance"]].set_index("Year")
            st.line_chart(chart_df)

            all_proj.append(proj_df)
            all_sched.append(sched_df)

    # Export all scenarios together
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
    st.info("Set your inputs, portfolio weights, frequencies, base RORs, and target goal, then click **Run projection**.")
