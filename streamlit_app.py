import streamlit as st
import pandas as pd
import numpy as np
import json

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


def blended_expected_return(you_ports, wife_ports):
    """
    Compute a simple blended expected return from all portfolios,
    weighted by their % weights (ignoring person split).
    """
    total_w = 0.0
    weighted_r = 0.0
    for p in you_ports + wife_ports:
        w = max(p["weight"], 0.0)
        total_w += w
        weighted_r += w * p["ror"]
    if total_w == 0:
        return 0.0
    return weighted_r / total_w  # in %


def run_monte_carlo(contrib_df, start_total_balance, mu_pct, sigma_pct, n_paths=1000):
    """
    Monte Carlo on combined ISA using:
      - mu_pct: expected return in %
      - sigma_pct: volatility in %

    Returns:
      years, paths (n_years x n_paths)
    """
    years = contrib_df["Year"].tolist()
    n_years = len(years)
    contribs = contrib_df["Combined_Annual_Contrib"].values

    mu = mu_pct / 100.0
    sigma = sigma_pct / 100.0

    paths = np.zeros((n_years, n_paths))

    for j in range(n_paths):
        v = start_total_balance
        for i in range(n_years):
            r = np.random.normal(mu, sigma)
            v = (v + contribs[i]) * (1 + r)
            paths[i, j] = v

    return years, paths

# -----------------------
# Streamlit UI
# -----------------------

st.set_page_config(page_title="Joint ISA Planner – Scenarios & Dynamic Portfolios", layout="wide")

st.title("Joint ISA Planner – Dynamic Portfolios & Base / Dip / Rise Scenarios")

st.markdown(
    """
This app helps you model **two ISAs (husband & wife)** over time, including:

- Starting balances and annual contributions  
- Annual contribution raises  
- ISA caps and optional surplus redirect between spouses  
- Up to **5 portfolios each** (different mixes, frequencies, returns)  
- Three market scenarios: **Base**, **Dip**, and **Rise**  
- A simple goal tracker (e.g. £1,000,000 target by a given year)  

It’s a **planning tool**, not financial advice. It doesn’t connect to live brokers or use live prices.
"""
)

with st.expander("ℹ️ How contributions, caps & raises work"):
    st.markdown(
        """
- You set each person's **initial annual contribution** in the start year.  
- Each year, that amount increases by **Annual Raise %**, *until* it hits that person's **ISA cap**.  
- If **surplus redirect** is ON and one spouse wants to contribute more than their cap:
  - Any extra can be automatically redirected into the **other spouse's ISA**, as long as they still have room under their cap.  
- Contributions are then **split across portfolios** according to your percentage weights.
        """
    )

with st.expander("ℹ️ How portfolios & returns work"):
    st.markdown(
        """
- Each spouse can define up to **5 portfolios** (P1–P5 / W1–W5).  
- For each portfolio you set:
  - A **label** (e.g. 'Core S&P', 'Tech', 'Global')  
  - A **% of that person's ISA**  
  - A **payment frequency** (Weekly, Biweekly, Monthly)  
  - A **base annual return %** (your assumption, e.g. 11.8%)  
- The app **normalises weights** so they don't need to sum exactly to 100%.  
- In each scenario:
  - **Base** uses your base RORs as you entered them.  
  - **Dip** subtracts a fixed number of percentage points from each portfolio's ROR.  
  - **Rise** adds that number of points.
        """
    )

with st.expander("ℹ️ How scenarios, goal & summary work"):
    st.markdown(
        """
- **Base / Dip / Rise** are not predictions, just *what-if* worlds:
  - Base: your central expectation.  
  - Dip: weaker markets (returns reduced by your chosen delta).  
  - Rise: stronger markets (returns increased by that delta).  

- **Goal**:  
  - You set a target value (e.g. £1,000,000).  
  - Each scenario's summary shows the **first year** where the combined ISA balance meets or exceeds that goal.  
  - If it never does within the chosen period, it will say **'Not reached'**.  

- **Quick Summary Box** (per scenario):
  - Final combined value  
  - Total deposits  
  - Total growth  
  - Approximate effective CAGR over the whole period  
  - Year the goal is reached (if at all)  
  - Average monthly funding required
        """
    )

# --------- PROFILE LOAD (must be before widgets that use defaults) ---------
if "profile" not in st.session_state:
    st.session_state["profile"] = {}
if "actuals_df" not in st.session_state:
    st.session_state["actuals_df"] = None

with st.sidebar:
    st.header("Profiles")

    uploaded_profile = st.file_uploader("Load profile JSON", type="json")
    if uploaded_profile is not None:
        try:
            profile_data = json.load(uploaded_profile)
            st.session_state["profile"] = profile_data
            st.success("Profile loaded. Settings below have been pre-filled.")
        except Exception as e:
            st.error(f"Could not load profile: {e}")

# Helper to read default from profile or fallback
def get_profile_value(key, default):
    return st.session_state["profile"].get(key, default)

with st.sidebar:
    st.header("Core Inputs")

    start_year = st.number_input(
        "Start Year",
        value=get_profile_value("start_year", 2026),
        step=1,
    )
    end_year = st.number_input(
        "Target End Year",
        value=get_profile_value("end_year", 2042),
        step=1,
        min_value=start_year,
    )

    st.subheader("Start Balances (£)")
    you_start_balance = st.number_input(
        "Start Balance (Husband)",
        value=get_profile_value("you_start_balance", 2513.0),
        step=100.0,
    )
    wife_start_balance = st.number_input(
        "Start Balance (Wife)",
        value=get_profile_value("wife_start_balance", 300.0),
        step=50.0,
    )

    st.subheader("Initial Annual Contributions (£)")
    st.caption("First-year annual contributions in the Start Year.")
    you_initial_contrib = st.number_input(
        "Initial Annual Contrib – Husband",
        value=get_profile_value("you_initial_contrib", 5096.0),
        step=100.0,
    )
    wife_initial_contrib = st.number_input(
        "Initial Annual Contrib – Wife",
        value=get_profile_value("wife_initial_contrib", 1200.0),
        step=100.0,
    )

    annual_raise_pct = st.number_input(
        "Annual Raise % (Before Cap)",
        value=get_profile_value("annual_raise_pct", 25.0),
        step=1.0,
        min_value=0.0,
        max_value=100.0,
    )

    st.subheader("Scenario Return Deltas (%)")
    st.caption("Each portfolio has a base ROR; Dip/Rise apply these deltas to all portfolios.")
    dip_delta = st.number_input(
        "Dip scenario: subtract (%)",
        value=get_profile_value("dip_delta", 3.0),
        step=0.5,
    )
    rise_delta = st.number_input(
        "Rise scenario: add (%)",
        value=get_profile_value("rise_delta", 3.0),
        step=0.5,
    )

    st.subheader("ISA Caps (£/yr)")
    you_cap = st.number_input(
        "Husband ISA Cap",
        value=get_profile_value("you_cap", 20000.0),
        step=1000.0,
    )
    wife_cap = st.number_input(
        "Wife ISA Cap",
        value=get_profile_value("wife_cap", 20000.0),
        step=1000.0,
    )

    redirect_surplus = st.checkbox(
        "Redirect surplus contributions to the other spouse once cap is hit?",
        value=get_profile_value("redirect_surplus", True),
    )

    st.subheader("Goal")
    target_goal = st.number_input(
        "Target portfolio value (£)",
        value=get_profile_value("target_goal", 1_000_000.0),
        step=50_000.0,
        min_value=0.0,
    )

    st.subheader("Frequencies (per year)")
    weeks_per_year = st.number_input(
        "Weeks per Year",
        value=get_profile_value("weeks_per_year", 52),
        step=1,
        min_value=1,
    )
    biweekly_per_year = st.number_input(
        "Biweekly Periods per Year",
        value=get_profile_value("biweekly_per_year", 26),
        step=1,
        min_value=1,
    )

# ------------- Portfolios -------------

saved_you_ports = st.session_state["profile"].get("you_ports", [])
saved_wife_ports = st.session_state["profile"].get("wife_ports", [])

st.markdown("### Husband Portfolios (up to 5)")
you_ports = []
cols_you = st.columns(5)
default_you_weights = [40.0, 30.0, 20.0, 5.0, 5.0]
default_you_freqs = ["Weekly", "Weekly", "Biweekly", "Monthly", "Monthly"]
default_you_rors = [11.8, 13.0, 9.0, 8.0, 7.0]

for i in range(5):
    saved = saved_you_ports[i] if i < len(saved_you_ports) else {}
    with cols_you[i]:
        label = st.text_input(
            f"H P{i+1} label",
            value=saved.get("label", f"P{i+1}"),
            key=f"you_label_{i}",
        )
        weight = st.number_input(
            f"{label} % of his ISA",
            value=float(saved.get("weight", default_you_weights[i])),
            step=5.0,
            key=f"you_weight_{i}",
        )
        freq_default = saved.get("freq", default_you_freqs[i])
        freq = st.selectbox(
            f"{label} frequency",
            options=["Weekly", "Biweekly", "Monthly"],
            index=["Weekly", "Biweekly", "Monthly"].index(freq_default),
            key=f"you_freq_{i}",
        )
        ror = st.number_input(
            f"{label} base ROR (%)",
            value=float(saved.get("ror", default_you_rors[i])),
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
    saved = saved_wife_ports[i] if i < len(saved_wife_ports) else {}
    with cols_w[i]:
        label = st.text_input(
            f"W P{i+1} label",
            value=saved.get("label", f"W{i+1}"),
            key=f"wife_label_{i}",
        )
        weight = st.number_input(
            f"{label} % of her ISA",
            value=float(saved.get("weight", default_wife_weights[i])),
            step=5.0,
            key=f"wife_weight_{i}",
        )
        freq_default = saved.get("freq", default_wife_freqs[i])
        freq = st.selectbox(
            f"{label} frequency",
            options=["Weekly", "Biweekly", "Monthly"],
            index=["Weekly", "Biweekly", "Monthly"].index(freq_default),
            key=f"wife_freq_{i}",
        )
        ror = st.number_input(
            f"{label} base ROR (%)",
            value=float(saved.get("ror", default_wife_rors[i])),
            step=0.5,
            key=f"wife_ror_{i}",
        )
        wife_ports.append({"label": label, "weight": weight, "freq": freq, "ror": ror})

# -------- Actual vs projected tracking setup --------
years_list = list(range(start_year, end_year + 1))
with st.expander("📈 Actual vs projected tracking (optional)"):
    st.markdown(
        """
Enter your **actual combined ISA balance** at each year-end as time goes on.  
The scenarios will then compare projected vs actual and tell you if you're **Ahead / Behind / On Track**.
        """
    )
    if (
        st.session_state["actuals_df"] is None
        or list(st.session_state["actuals_df"].get("Year", [])) != years_list
    ):
        st.session_state["actuals_df"] = pd.DataFrame(
            {
                "Year": years_list,
                "Actual_Combined_Balance": [None] * len(years_list),
            }
        )

    edited_actuals = st.data_editor(
        st.session_state["actuals_df"],
        num_rows="fixed",
        use_container_width=True,
        key="actuals_editor",
    )
    st.session_state["actuals_df"] = edited_actuals

# -------- Monte Carlo risk-band settings --------
with st.expander("🎲 Risk band / Monte Carlo (Base scenario only)", expanded=False):
    st.markdown(
        """
This adds a **simulated range of outcomes** around your Base scenario, using a simple Monte Carlo model.

- It uses a blended expected return based on your portfolio base RORs.  
- You choose an assumed **annual volatility**.  
- It simulates many random paths and shows:
  - Percentile band (5th–95th) over time  
  - Probability of hitting your goal by the end year  
        """
    )
    mc_enabled = st.checkbox("Run Monte Carlo risk band on Base scenario?", value=False)
    mc_volatility = st.number_input("Assumed annual volatility %", value=15.0, step=1.0, min_value=0.0)
    mc_paths = st.number_input("Number of Monte Carlo paths", value=1000, step=100, min_value=100)

run_btn = st.button("Run projection")

# -------- PROFILE SAVE BUTTON --------
profile = {
    "start_year": start_year,
    "end_year": end_year,
    "you_start_balance": you_start_balance,
    "wife_start_balance": wife_start_balance,
    "you_initial_contrib": you_initial_contrib,
    "wife_initial_contrib": wife_initial_contrib,
    "annual_raise_pct": annual_raise_pct,
    "dip_delta": dip_delta,
    "rise_delta": rise_delta,
    "you_cap": you_cap,
    "wife_cap": wife_cap,
    "redirect_surplus": redirect_surplus,
    "target_goal": target_goal,
    "weeks_per_year": weeks_per_year,
    "biweekly_per_year": biweekly_per_year,
    "you_ports": you_ports,
    "wife_ports": wife_ports,
}

profile_json = json.dumps(profile, indent=2).encode("utf-8")
st.download_button(
    "💾 Download current profile",
    data=profile_json,
    file_name="isa_profile.json",
    mime="application/json",
)

if run_btn:
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

    actual_df = st.session_state["actuals_df"].copy()
    if "Actual_Combined_Balance" in actual_df.columns:
        actual_df["Actual_Combined_Balance"] = pd.to_numeric(
            actual_df["Actual_Combined_Balance"], errors="coerce"
        )

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

            # Actual vs projected comparison
            st.markdown("**Actual vs projected (if actual data entered)**")

            merged = proj_df.merge(actual_df, on="Year", how="left")
            tol = 0.05  # 5% tolerance band

            def classify_status(row):
                actual = row["Actual_Combined_Balance"]
                proj = row["Combined_End_Balance"]
                if pd.isna(actual):
                    return ""
                if actual > proj * (1 + tol):
                    return "Ahead"
                if actual < proj * (1 - tol):
                    return "Behind"
                return "On Track"

            merged["Status"] = merged.apply(classify_status, axis=1)
            merged["Diff"] = merged["Actual_Combined_Balance"] - merged["Combined_End_Balance"]

            display_cols = [
                "Year",
                "Combined_End_Balance",
                "Actual_Combined_Balance",
                "Diff",
                "Status",
            ]

            st.dataframe(
                merged[display_cols].style.format(
                    {
                        "Combined_End_Balance": "£{:,.0f}",
                        "Actual_Combined_Balance": "£{:,.0f}",
                        "Diff": "£{:,.0f}",
                    }
                ),
                use_container_width=True,
            )

            if merged["Actual_Combined_Balance"].notna().any():
                chart_compare = merged.set_index("Year")[["Combined_End_Balance", "Actual_Combined_Balance"]]
                st.line_chart(chart_compare)

            # Monte Carlo risk band (Base scenario only)
            if scenario_name == "Base" and mc_enabled and mc_volatility > 0 and mc_paths > 0:
                st.markdown("### 🎲 Monte Carlo risk band (Base scenario)")

                blended_mu = blended_expected_return(you_ports, wife_ports)
                start_total_balance = you_start_balance + wife_start_balance

                years_mc, paths = run_monte_carlo(
                    contrib_df,
                    start_total_balance=start_total_balance,
                    mu_pct=blended_mu,
                    sigma_pct=mc_volatility,
                    n_paths=int(mc_paths),
                )

                # Percentiles per year
                percentiles = {
                    "P5": np.percentile(paths, 5, axis=1),
                    "P25": np.percentile(paths, 25, axis=1),
                    "P50": np.percentile(paths, 50, axis=1),
                    "P75": np.percentile(paths, 75, axis=1),
                    "P95": np.percentile(paths, 95, axis=1),
                }

                mc_df = pd.DataFrame(
                    {"Year": years_mc, **percentiles}
                )

                # Probability of hitting goal by end
                if target_goal > 0:
                    final_vals = paths[-1, :]
                    prob_hit = (final_vals >= target_goal).mean() * 100
                else:
                    prob_hit = None

                st.info(
                    f"""
**Blended expected return (used for MC):** {blended_mu:.2f}%  
**Assumed volatility:** {mc_volatility:.2f}%  
**Simulated paths:** {int(mc_paths)}  
**Probability of hitting goal (£{target_goal:,.0f}) by {end_year}:** {prob_hit:.1f}%  
                    """
                    if prob_hit is not None
                    else f"""
**Blended expected return (used for MC):** {blended_mu:.2f}%  
**Assumed volatility:** {mc_volatility:.2f}%  
**Simulated paths:** {int(mc_paths)}  
(Goal not set or zero, so probability not calculated.)
                    """
                )

                st.markdown("**Monte Carlo percentile band (Base scenario)**")
                st.dataframe(
                    mc_df.style.format(
                        {c: "£{:,.0f}" for c in mc_df.columns if c != "Year"}
                    ),
                    use_container_width=True,
                )

                mc_chart_df = mc_df.set_index("Year")
                st.line_chart(mc_chart_df)

            # Projected balance chart
            st.markdown("**Projected combined ISA balance over time**")
            chart_df = proj_df[["Year", "Combined_End_Balance"]].set_index("Year")
            st.line_chart(chart_df)

            all_proj.append(proj_df)
            all_sched.append(sched_df)

    # Export all scenarios together
    st.subheader("Export all scenarios")
    proj_concat = pd.concat(all_proj, ignore_index=True)
    sched_concat = pd.concat(all_sched, ignore_index=True)
    merged_all = proj_concat.merge(sched_concat, on=["Scenario", "Year"], how="left")

    csv = merged_all.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV (all scenarios + schedules)",
        data=csv,
        file_name="joint_isa_scenarios_dynamic_portfolios.csv",
        mime="text/csv",
    )

    st.markdown(
        """
        ---
        **Disclaimer:** This app is for educational and planning purposes only and does not constitute financial advice.  
        Please do your own research or speak to a qualified adviser before making investment decisions.
        """
    )
else:
    st.info("Set your inputs, portfolio weights, frequencies, base RORs, and target goal, then click **Run projection**.")
