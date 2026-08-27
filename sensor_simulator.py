import pandas as pd
import numpy as np
import joblib
import time

# Load dataset
df = pd.read_csv("ai4i2020.csv")
df.columns = df.columns.str.strip()

# Load trained model and encoder
model = joblib.load("predictive_maintenance_model.pkl")
encoder = joblib.load("type_encoder.pkl")

# Select 10 machines
machines = df.sample(10, random_state=42).copy()

# Give each machine an ID
machines["Machine ID"] = [
    f"M-{i:02d}" for i in range(1, 11)
]

# Display initial machines
print(
    machines[
        [
            "Machine ID",
            "Type",
            "Air temperature",
            "Process temperature",
            "Rotational speed",
            "Torque",
            "Tool wear",
            "Machine failure"
        ]
    ]
)

# Sensor columns
sensor_columns = [
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear"
]

# Display dataset statistics
print(df[sensor_columns].describe())

# Analyze average sensor values for failed/non-failed machines
failure_analysis = df.groupby(
    "Machine failure"
)[sensor_columns].mean()

print("\nAverage sensor values:")
print(failure_analysis)


# Simulate the next sensor state
def simulate_next_state(machine, condition):

    next_state = machine.copy()

    if condition == "healthy":

        next_state["Air temperature"] += np.random.normal(
            0, 0.15
        )

        next_state["Process temperature"] += np.random.normal(
            0, 0.10
        )

        next_state["Rotational speed"] += np.random.normal(
            0, 15
        )

        next_state["Torque"] += np.random.normal(
            0, 1
        )

        next_state["Tool wear"] += np.random.uniform(
            0.5, 1.5
        )

    elif condition == "deteriorating":

        next_state["Air temperature"] += np.random.normal(
            0.15, 0.15
        )

        next_state["Process temperature"] += np.random.normal(
            0.10, 0.10
        )

        next_state["Rotational speed"] -= np.random.normal(
            8, 5
        )

        next_state["Torque"] += np.random.normal(
            1.0, 0.5
        )

        next_state["Tool wear"] += np.random.uniform(
            1.5, 3
        )

    elif condition == "critical":

        next_state["Air temperature"] += np.random.normal(
            0.25, 0.15
        )

        next_state["Process temperature"] += np.random.normal(
            0.15, 0.10
        )

        next_state["Rotational speed"] -= np.random.normal(
            15, 7
        )

        next_state["Torque"] += np.random.normal(
            1.5, 0.7
        )

        next_state["Tool wear"] += np.random.uniform(
            2, 4
        )

    return next_state


# Machine conditions
conditions = [
    "healthy",        # M-01
    "healthy",        # M-02
    "healthy",        # M-03
    "healthy",        # M-04
    "healthy",        # M-05
    "deteriorating",  # M-06
    "healthy",        # M-07
    "healthy",        # M-08
    "healthy",        # M-09
    "healthy"         # M-10
]


# Keep track of the latest state of every machine
current_machines = machines.copy()


# Run 10 simulation cycles
for cycle in range(10):

    results = []

    # Process every machine
    for i in range(len(current_machines)):

        machine = current_machines.iloc[i]

        # Generate next sensor state
        next_machine = simulate_next_state(
            machine,
            conditions[i]
        )

        # Save the new state
        current_machines.iloc[i] = next_machine

        # Select data required by XGBoost
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

        # Encode machine Type
        prediction_data["Type"] = encoder.transform(
            [prediction_data["Type"]]
        )[0]

        # Make sure all values are numeric
        prediction_data = prediction_data.astype(float)

        # Predict failure
        prediction = model.predict(
            prediction_data.to_frame().T
        )[0]

        # Get probability of failure
        probability = model.predict_proba(
            prediction_data.to_frame().T
        )[0][1]

        # Store result
        results.append(
            {
                "Machine ID": machine["Machine ID"],
                "Condition": conditions[i],
                "Failure Probability": probability,
                "Prediction": prediction
            }
        )

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Display current time step
    print("\n" + "=" * 60)
    print(f"TIME STEP {cycle + 1}")
    print("=" * 60)

    print(
        results_df.to_string(index=False)
    )

    # Wait 2 seconds before next update
    time.sleep(2)