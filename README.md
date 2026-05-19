# Oakwood Capital — Strategy Platform

Multi-strategy backtest platform built with Streamlit. Two strategies hosted under one URL:

1. **SMI Income meets Digital Assets** — SMI 20 with BTC overlay
2. **Swiss Dividend Income + Bitcoin** — Top 10 yield + BTC

## Folder Structure

```
oakwood-platform/
├── Home.py                              ← Landing page
├── pages/
│   ├── 1_SMI_Strategy.py               ← SMI 20 strategy
│   └── 2_Top_10_Dividend_Strategy.py   ← Top 10 yield strategy
├── assets/oakwood_logo.png
├── .streamlit/config.toml              ← Theme
├── requirements.txt
└── README.md
```

## Local Run

```bash
pip install -r requirements.txt
streamlit run Home.py
```

## Notes

- Public Streamlit Cloud requires public GitHub repo.
- Yahoo Finance data is not licensed for commercial use.
- Free tier: 1 GB RAM, sleeps after ~7 days inactivity.
- For illustrative purposes only — not investment advice.
