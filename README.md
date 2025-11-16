
# **Joint ISA Planner – Dynamic Portfolio Scenario Simulator**

A powerful interactive tool for projecting long-term ISA growth for a **two-adult household**, including:

* ISA contribution automation
* Annual raises
* ISA caps
* Portfolio allocation planning
* Per-portfolio return assumptions
* Weekly / biweekly / monthly investment scheduling
* Base, Dip, and Rise performance scenarios

Built with **Streamlit** + **Pandas**.

---

## **Features**

### **Household ISA Modelling**

* Husband and Wife ISA accounts
* Independent ISA caps (default £20,000)
* Annual contribution raises
* Optional “redirect husband’s surplus to wife” logic
* Start balance + initial annual contributions

---

### **Flexible Portfolio Structure**

Each person can define **up to 5 portfolios**:

* Custom portfolio label
* Weight as % of their ISA
* Payment frequency:

  * Weekly
  * Biweekly
  * Monthly
* Base annual return (%) per portfolio (e.g. Tech = 14%, Global = 10%, Bonds = 5%)

The app automatically normalises weights and splits contributions accordingly.

---

### **Multiple Scenarios**

For stress-testing portfolio outcomes:

* **Base Scenario** – uses each portfolio’s base ROR
* **Dip Scenario** – all RORs reduced by X%
* **Rise Scenario** – all RORs increased by Y%

Each scenario runs independently and produces its own projection.

---

### **Detailed Outputs Per Scenario**

* Year-by-year:

  * Annual contributions
  * Monthly funding required
  * Start and end balances
  * Combined ISA value
  * Cumulative deposits and growth

* Per-portfolio contributions per period:

  * Weekly
  * Biweekly
  * Monthly

* Growth chart

* CSV export

---

### **Quick Summary Box**

Each scenario displays:

* Final combined value
* Total deposits
* Total growth
* Effective CAGR
* Year £1M reached (if reached)
* Average monthly funding required

---

# **How to Use the App**

## 1. **Install Dependencies**

```bash
pip install streamlit pandas
```

## 2. **Run the App**

```bash
streamlit run app.py
```

Your browser will open automatically at:

```
http://localhost:8501
```

---

# **User Guide**

## **Step 1 — Enter Core ISA Inputs**

In the sidebar:

* Start Year / End Year
* Husband & Wife starting balances
* Initial annual contributions
* “Annual Raise %” applied each January
* ISA Caps
* “Redirect surplus to wife” toggle
* Frequencies per year (weeks, biweeks)

---

## **Step 2 — Configure Portfolios**

For each of the 5 portfolio slots (you can leave unused ones at 0%):

* Name the portfolio
* Assign % of ISA
* Choose Weekly / Biweekly / Monthly schedule
* Set its **base annual return %**

Example:

| Portfolio | %   | Freq     | ROR   |
| --------- | --- | -------- | ----- |
| Core S&P  | 50% | Weekly   | 11.5% |
| Tech 5    | 30% | Biweekly | 14%   |
| Global    | 20% | Monthly  | 9%    |

---

## **Step 3 — Configure Scenario Performance**

* Dip scenario: subtract X% from all portfolio RORs
* Rise scenario: add Y% to all RORs

---

## **Step 4 — Run Projection**

Click **Run**.
You’ll see 3 tabs:

* **Base**
* **Dip**
* **Rise**

Each has:

1. A **Quick Summary Box**
2. Year-by-year projections
3. Per-portfolio schedules
4. Growth chart

---

## **Step 5 — Export Your Results**

At the bottom, export a **CSV file** containing:

* All scenarios
* All year-by-year projections
* All per-portfolio contribution schedules

---

# **Project Structure**

```
project/
  ├── app.py
  ├── README.md
  └── requirements.txt
```

`requirements.txt` (optional):

```
streamlit
pandas
```

---

# Notes & Tips

* You do **not** need to ensure your portfolio weights add to 100% — the app normalises them.
* You can model any number of portfolios up to 5, just set unused ones to 0%.
* You can model long-term goals:

  * £1M
  * £2M
  * Retirement timelines
* The Quick Summary Box makes it easy to compare Base / Dip / Rise futures.
