import numpy as np
import pandas as pd

# Define the data for each run
data = [
    {"ASR": 50.36, "SAQ": 7.33, "TAQ": 36.53, "ER": 12.41, "Total": 411, "D_sql": 1.81490853249783, "D_xss": 2.3975647615912914},
    {"ASR": 48.90, "SAQ": 8.73, "TAQ": 41.86, "ER": 12.15, "Total": 362, "D_sql": 1.9599634884623869, "D_xss": 2.4212366456914323},
    {"ASR": 45.53, "SAQ": 6.52, "TAQ": 43.52, "ER": 10.37, "Total": 347, "D_sql": 1.7240545390556907, "D_xss": 1.8924568651889142},
    {"ASR": 51.83, "SAQ": 5.54, "TAQ": 37.79, "ER": 10.27, "Total": 409, "D_sql": 1.7033654478284377, "D_xss": 1.5529362289699344},
    {"ASR": 50.10, "SAQ": 7.62, "TAQ": 41.64, "ER": 11.91, "Total": 487, "D_sql": 1.7600287834507922, "D_xss": 2.3990253507596275},
    {"ASR": 50.14, "SAQ": 6.05, "TAQ": 41.10, "ER": 12.53, "Total": 367, "D_sql": 1.815139340204159, "D_xss": 2.5322936254982156},
    {"ASR": 50.79, "SAQ": 6.37, "TAQ": 40.30, "ER": 9.74, "Total": 380, "D_sql": 1.75673233951125, "D_xss": 2.4081996660884544},
    {"ASR": 44.96, "SAQ": 5.44, "TAQ": 44.00, "ER": 10.31, "Total": 456, "D_sql": 1.6440350878804981, "D_xss": 2.2580966343028632},
    {"ASR": 46.97, "SAQ": 8.78, "TAQ": 43.61, "ER": 12.99, "Total": 462, "D_sql": 1.8237873777895812, "D_xss": 2.565453369230267},
    {"ASR": 49.09, "SAQ": 7.49, "TAQ": 41.08, "ER": 11.16, "Total": 493, "D_sql": 1.694087485485543, "D_xss": 1.944876217690724}
]

# Calculate Nsucc and Dsucc for each run
for entry in data:
    entry["Nsucc"] = entry["ASR"] / 100 * entry["Total"]
    entry["Dsucc"] = entry["D_sql"] * 0.9 + entry["D_xss"] * 0.1

# Create DataFrame
df = pd.DataFrame(data)

# Compute mean, median, and coefficient of variation (CV = std / mean)
metrics = ["ASR", "SAQ", "TAQ", "ER", "Nsucc", "Dsucc"]
stats = {
    "Mean": df[metrics].mean(),
    "Median": df[metrics].median(),
    "CV": df[metrics].std() / df[metrics].mean()
}

# Convert to DataFrame for display
result_df = pd.DataFrame(stats)
import ace_tools as tools; tools.display_dataframe_to_user(name="Attack Metrics Summary", dataframe=result_df)

