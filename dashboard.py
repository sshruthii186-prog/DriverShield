
import streamlit as st
import pandas as pd
import os

# Page configuration
st.set_page_config(
    page_title="DriverShield Dashboard",
    page_icon="🚗",
    layout="wide"
)

# Dashboard heading
st.title("🚗 DriverShield")
st.subheader("Driver Monitoring Dashboard")
st.write(
    "Monitor recorded drowsiness, possible yawning, "
    "and possible head-turn events."
)

st.divider()

# Locate CSV file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "drowsiness_log.csv")

if not os.path.exists(CSV_FILE):
    st.warning(
        "No event log found. Run eye_detection.py first."
    )

else:
    try:
        df = pd.read_csv(CSV_FILE)

        if df.empty:
            st.info("No events recorded yet.")

        elif "Event" not in df.columns:
            st.error("The CSV file is missing the Event column.")

        else:
            # Clean event names
            df["Event"] = df["Event"].astype(str).str.strip()

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

            total_events = len(df)

            # Overview metrics
            st.header("📊 Event Overview")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Total Events", total_events)
            col2.metric("Drowsiness Alerts", drowsiness_count)
            col3.metric("Possible Yawns", yawn_count)
            col4.metric("Possible Head Turns", head_turn_count)

            st.divider()

            # Event summary chart
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

            st.bar_chart(
                summary.set_index("Event Type")
            )

            st.divider()

            # Event history
            st.header("📋 Event History")

            display_df = df.copy()

            if "Timestamp" in display_df.columns:
                display_df = display_df.sort_values(
                    "Timestamp",
                    ascending=False
                )

            st.dataframe(
                display_df,
                hide_index=True,
                width="stretch"
            )

            # Download event history
            st.download_button(
                label="📥 Download Event History",
                data=df.to_csv(index=False),
                file_name="drivershield_event_history.csv",
                mime="text/csv"
            )

            st.divider()

            st.caption(
                "Prototype monitoring only. These detections "
                "are estimates, not confirmed evidence of "
                "drowsiness or driver distraction."
            )

            if st.button("🔄 Refresh Dashboard"):
                st.rerun()

    except (pd.errors.ParserError, UnicodeDecodeError, OSError) as error:
        st.error("Unable to read drowsiness_log.csv.")
        st.write(str(error))

