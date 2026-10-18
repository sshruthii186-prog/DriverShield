import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="DriverShield Demo",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 DriverShield")
st.subheader("Driver Monitoring Dashboard")

st.write(
    "An AI-based prototype for monitoring possible "
    "driver drowsiness and distraction."
)

st.warning(
    "DEMO MODE: The records below are illustrative sample "
    "data, not live webcam detections."
)

# Illustrative sample records for the online demonstration
df = pd.DataFrame({
    "Timestamp": [
        "2026-10-18 09:10:00",
        "2026-10-18 09:12:00",
        "2026-10-18 09:15:00",
        "2026-10-18 09:20:00",
        "2026-10-18 09:25:00",
        "2026-10-18 09:30:00"
    ],
    "Event": [
        "Drowsiness Alert",
        "Possible Yawn",
        "Possible Head Turn",
        "Drowsiness Alert",
        "Possible Yawn",
        "Possible Head Turn"
    ],
    "Duration": [2.5, 1.8, 1.2, 3.0, 2.1, 1.5]
})

# Event counts
events = df["Event"]

drowsiness_count = int(
    (events == "Drowsiness Alert").sum()
)

yawn_count = int(
    (events == "Possible Yawn").sum()
)

head_turn_count = int(
    (events == "Possible Head Turn").sum()
)

st.divider()
st.header("📊 Event Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Events", len(df))
col2.metric("Drowsiness Alerts", drowsiness_count)
col3.metric("Possible Yawns", yawn_count)
col4.metric("Possible Head Turns", head_turn_count)

st.divider()
st.header("📈 Event Summary")

summary = pd.DataFrame({
    "Event Type": [
        "Drowsiness Alert",
        "Possible Yawn",
        "Possible Head Turn"
    ],
    "Count": [
        drowsiness_count,
        yawn_count,
        head_turn_count
    ]
})

st.bar_chart(summary.set_index("Event Type"))

st.divider()
st.header("📋 Event History")

st.dataframe(df, hide_index=True)

st.download_button(
    label="📥 Download Demo Event History",
    data=df.to_csv(index=False),
    file_name="drivershield_demo_events.csv",
    mime="text/csv"
)

st.caption(
    "Educational prototype only. These sample records "
    "do not represent validated real-world detections."
)