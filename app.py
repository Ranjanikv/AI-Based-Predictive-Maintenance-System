import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Predictive Maintenance Monitor",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("AI PREDICTIVE MAINTENANCE MONITOR")
st.subheader("Real-Time Machine Monitoring")


# ============================================================
# LOAD DATA AND MODEL
# ============================================================

df = pd.read_csv("ai4i2020.csv")

df.columns = df.columns.str.strip()

model = joblib.load(
    "predictive_maintenance_model.pkl"
)

encoder = joblib.load(
    "type_encoder.pkl"
)


# ============================================================
# RESET SIMULATION
# ============================================================

if st.button("Reset Simulation"):

    st.session_state.clear()

    st.rerun()


# ============================================================
# INITIALIZE MACHINES
# ============================================================

if "machines" not in st.session_state:

    machines = df.sample(
        10,
        random_state=42
    ).copy()

    machines["Machine ID"] = [
        f"M-{i:02d}"
        for i in range(1, 11)
    ]

    st.session_state.machines = machines


# ============================================================
# MACHINE CONDITIONS
# ============================================================

conditions = [
    "healthy",          # M-01
    "healthy",          # M-02
    "healthy",          # M-03
    "healthy",          # M-04
    "healthy",          # M-05
    "deteriorating",    # M-06
    "healthy",          # M-07
    "healthy",          # M-08
    "healthy",          # M-09
    "healthy"           # M-10
]


# ============================================================
# SHARED MACHINE COLORS
# ============================================================

machine_colors = {
    "M-01": "#1f77b4",
    "M-02": "#8ecae6",
    "M-03": "#2ca02c",
    "M-04": "#ff7f0e",
    "M-05": "#9467bd",
    "M-06": "#da5b5b",
    "M-07": "#17becf",
    "M-08": "#bcbd22",
    "M-09": "#e377c2",
    "M-10": "#7f7f7f"
}


# ============================================================
# INITIALIZE TIME STEP
# ============================================================

if "time_step" not in st.session_state:

    st.session_state.time_step = 0


# ============================================================
# INITIALIZE HISTORY
# ============================================================

if "history" not in st.session_state:

    st.session_state.history = []


# ============================================================
# SENSOR SIMULATION
# ============================================================

def simulate_next_state(machine, condition):

    next_state = machine.copy()

    # --------------------------------------------------------
    # HEALTHY
    # --------------------------------------------------------

    if condition == "healthy":

        next_state["Air temperature"] += np.random.normal(
            0,
            0.15
        )

        next_state["Process temperature"] += np.random.normal(
            0,
            0.10
        )

        next_state["Rotational speed"] += np.random.normal(
            0,
            15
        )

        next_state["Torque"] += np.random.normal(
            0,
            1
        )

        next_state["Tool wear"] += np.random.uniform(
            0.5,
            1.5
        )


    # --------------------------------------------------------
    # DETERIORATING
    # --------------------------------------------------------

    elif condition == "deteriorating":

        next_state["Air temperature"] += np.random.normal(
            0.25,
            0.10
        )

        next_state["Process temperature"] += np.random.normal(
            0.18,
            0.08
        )

        next_state["Rotational speed"] -= np.random.normal(
            12,
            4
        )

        next_state["Torque"] += np.random.normal(
            1.5,
            0.4
        )

        next_state["Tool wear"] += np.random.uniform(
            2,
            4
        )


    # --------------------------------------------------------
    # CRITICAL
    # --------------------------------------------------------

    elif condition == "critical":

        next_state["Air temperature"] += np.random.normal(
            0.30,
            0.10
        )

        next_state["Process temperature"] += np.random.normal(
            0.20,
            0.08
        )

        next_state["Rotational speed"] -= np.random.normal(
            18,
            5
        )

        next_state["Torque"] += np.random.normal(
            2,
            0.5
        )

        next_state["Tool wear"] += np.random.uniform(
            3,
            5
        )


    return next_state


# ============================================================
# RUN ONE TIME STEP
# ============================================================

st.session_state.time_step += 1

current_machines = st.session_state.machines

results = []


for i in range(len(current_machines)):

    machine = current_machines.iloc[i]

    next_machine = simulate_next_state(
        machine,
        conditions[i]
    )


    # --------------------------------------------------------
    # UPDATE MACHINE STATE
    # --------------------------------------------------------

    st.session_state.machines.iloc[i] = next_machine


    # --------------------------------------------------------
    # PREPARE DATA FOR XGBOOST
    # --------------------------------------------------------

    prediction_data = next_machine[
        [
            "Type",
            "Air temperature",
            "Process temperature",
            "Rotational speed",
            "Torque",
            "Tool wear"
        ]
    ].copy()


    # Encode machine type

    prediction_data["Type"] = encoder.transform(
        [prediction_data["Type"]]
    )[0]


    # Make all values numeric

    prediction_data = prediction_data.astype(float)


    # --------------------------------------------------------
    # XGBOOST PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(
        prediction_data.to_frame().T
    )[0]


    probability = model.predict_proba(
        prediction_data.to_frame().T
    )[0][1]


    # --------------------------------------------------------
    # STORE RESULT
    # --------------------------------------------------------

    results.append({

        "Machine ID":
            next_machine["Machine ID"],

        "Condition":
            conditions[i],

        "Failure Probability":
            probability * 100,

        "Prediction":
            prediction

    })


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# SAVE SENSOR HISTORY
# ============================================================

for i in range(len(st.session_state.machines)):

    machine = st.session_state.machines.iloc[i]

    result = results_df.iloc[i]


    st.session_state.history.append({

        "Time":
            st.session_state.time_step,

        "Machine ID":
            machine["Machine ID"],

        "Failure Probability":
            result["Failure Probability"],

        "Air Temperature":
            machine["Air temperature"],

        "Process Temperature":
            machine["Process temperature"],

        "RPM":
            machine["Rotational speed"],

        "Torque":
            machine["Torque"],

        "Tool Wear":
            machine["Tool wear"]

    })


# ============================================================
# HISTORY DATAFRAME
# ============================================================

history_df = pd.DataFrame(
    st.session_state.history
)


# ============================================================
# MACHINE STATUS
# ============================================================

def get_status(probability):

    if probability >= 60:

        return "Critical"

    elif probability >= 30:

        return "At Risk"

    else:

        return "Healthy"


results_df["Status"] = results_df[
    "Failure Probability"
].apply(get_status)


# ============================================================
# KPI CALCULATIONS
# ============================================================

total_machines = len(
    results_df
)

at_risk = (
    results_df["Failure Probability"] >= 30
).sum()

healthy = total_machines - at_risk


# ============================================================
# KPI CARDS
# ============================================================

st.divider()

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Total Machines",
        total_machines
    )


with col2:

    st.metric(
        "Healthy",
        healthy
    )


with col3:

    st.metric(
        "At Risk",
        at_risk
    )


with col4:

    st.metric(
        "Time Step",
        st.session_state.time_step
    )


st.divider()


# ============================================================
# GRAPHS
# ============================================================

col1, col2 = st.columns(2)


# ============================================================
# BAR GRAPH
# ============================================================

with col1:

    st.subheader(
        "Current Failure Probability"
    )


    bar_data = results_df.sort_values(
        "Failure Probability",
        ascending=False
    )


    fig1, ax1 = plt.subplots(
        figsize=(8, 5)
    )


    # Use the same machine colors as Plotly

    bar_colors = [
        machine_colors[machine_id]
        for machine_id in bar_data["Machine ID"]
    ]


    ax1.bar(
        bar_data["Machine ID"],
        bar_data["Failure Probability"],
        color=bar_colors
    )


    ax1.set_xlabel(
        "Machine"
    )


    ax1.set_ylabel(
        "Failure Probability (%)"
    )


    ax1.set_title(
        "Current Machine Risk"
    )


    ax1.set_ylim(
        0,
        max(
            10,
            bar_data["Failure Probability"].max() + 10
        )
    )


    ax1.grid(
        axis="y",
        linestyle="--",
        alpha=0.3
    )


    plt.xticks(
        rotation=45
    )


    fig1.tight_layout()


    st.pyplot(
        fig1,
        use_container_width=True
    )


# ============================================================
# PLOTLY LINE GRAPH
# ============================================================

with col2:

    st.subheader(
        "Failure Probability Over Time"
    )


    fig2 = go.Figure()


    for machine_id in history_df[
        "Machine ID"
    ].unique():

        machine_history = history_df[
            history_df["Machine ID"] == machine_id
        ]


        fig2.add_trace(
            go.Scatter(

                x=machine_history["Time"],

                y=machine_history[
                    "Failure Probability"
                ],

                mode="lines+markers",

                name=machine_id,

                line=dict(
                    color=machine_colors[machine_id]
                ),

                marker=dict(
                    color=machine_colors[machine_id]
                ),

                customdata=machine_history[
                    [
                        "Air Temperature",
                        "Process Temperature",
                        "RPM",
                        "Torque",
                        "Tool Wear"
                    ]
                ].values,

                hovertemplate=
                    "Air Temp: %{customdata[0]:.2f}<br>"
                    "Process Temp: %{customdata[1]:.2f}<br>"
                    "RPM: %{customdata[2]:.0f}<br>"
                    "Torque: %{customdata[3]:.2f}<br>"
                    "Tool Wear: %{customdata[4]:.2f}"
                    "<extra></extra>"
            )
        )


    fig2.update_layout(

        height=500,

        xaxis_title=
            "Time Step",

        yaxis_title=
            "Failure Probability (%)",

        title=
            "Machine Risk Trend",

        hovermode=
            "closest",

        margin=dict(
            l=60,
            r=20,
            t=60,
            b=50
        )
    )


    st.plotly_chart(
        fig2,
        use_container_width=True
    )


# ============================================================
# MACHINE STATUS TABLE
# ============================================================

st.divider()

st.subheader(
    "Machine Status"
)


status_table = results_df[
    [
        "Machine ID",
        "Failure Probability",
        "Status"
    ]
].copy()


status_table[
    "Failure Probability"
] = status_table[
    "Failure Probability"
].round(2)


status_table = status_table.sort_values(
    "Failure Probability",
    ascending=False
)


st.dataframe(
    status_table,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# CURRENT SENSOR READINGS
# ============================================================

st.divider()

st.subheader(
    "Current Sensor Readings"
)


sensor_table = st.session_state.machines[
    [
        "Machine ID",
        "Type",
        "Air temperature",
        "Process temperature",
        "Rotational speed",
        "Torque",
        "Tool wear"
    ]
].copy()


st.dataframe(
    sensor_table,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# AUTO REFRESH
# ============================================================

time.sleep(3)

st.rerun()