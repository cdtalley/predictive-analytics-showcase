"""Streamlit stub: product KPI tree over synthetic events."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

DATA = Path(__file__).resolve().parents[1] / "data" / "synthetic" / "product_events.csv"

st.set_page_config(page_title="Product KPI Showcase", layout="wide")
st.title("Product analytics (synthetic)")
st.caption("Demo surface only. Regenerable dummy events, not client data.")

if not DATA.exists():
    st.error("Run `python -m src.generate_all` first.")
    st.stop()

df = pd.read_csv(DATA)
cohort = st.sidebar.multiselect("Cohort", sorted(df["cohort"].unique()), default=sorted(df["cohort"].unique()))
plan = st.sidebar.multiselect("Plan", sorted(df["plan"].unique()), default=sorted(df["plan"].unique()))
view = df[df["cohort"].isin(cohort) & df["plan"].isin(plan)]

signups = view[view["event"] == "signup"]["user_id"].nunique()
activates = view[view["event"] == "activate"]["user_id"].nunique()
paid = view[view["event"] == "paid_convert"]["user_id"].nunique()

c1, c2, c3 = st.columns(3)
c1.metric("Signups", signups)
c2.metric("Activated", activates, f"{activates / max(signups, 1):.0%} of signup")
c3.metric("Paid convert", paid, f"{paid / max(activates, 1):.0%} of activated")

st.subheader("Event volume")
st.bar_chart(view.groupby("event")["value"].sum())
