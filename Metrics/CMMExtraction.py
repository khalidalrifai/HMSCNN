import re
import pandas as pd


def extract_values_from_file(file_path):
    keywords = [
        "Condition Positive (P):",
        "Condition Negative (N):",
        "True Positive (TP):",
        "True Negative (TN):",
        "False Positive (FP):",
        "False Negative (FN):",
        "Sensitivity, Recall, Hit Rate, or True Positive Rate (TPR):",
        "Specificity, Selectivity, or True Negative Rate (TNR):",
        "Precision or Positive Predictive Value (PPV):",
        "Negative Predictive Value (NPV):",
        "Miss Rate or False Negative Rate (FNR):",
        "Fall-Out or False Positive Rate (FPR):",
        "False Discovery Rate (FDR):",
        "False Omission Rate (FOR):",
        "Positive Likelihood Ratio (LR+):",
        "Negative Likelihood Ratio (LR-):",
        "Prevalence Threshold (PT):",
        "Threat Score (TS) or Critical Success Index (CSI):",
        "Prevalence:",
        "Accuracy (ACC):",
        "Balanced Accuracy (BA):",
        "The Harmonic Mean of Precision and Sensitivity (F One Score):",
        "Matthews Correlation Coefficient (MCC):",
        "Fowlkes Mallows Index (FM):",
        "Informedness or Bookmaker Informedness (BM):",
        "Markedness (MK):",
        "Diagnostic Odds Ratio (DOR):"
    ]

    # Read the file
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Extract the values for the keywords
    extracted_values = {}
    for line in lines:
        for keyword in keywords:
            if keyword in line:
                # Extract the numerical value using regex
                match = re.search(r"[-+]?\d*\.\d+|\d+", line)
                if match:
                    extracted_values[keyword] = float(match.group())

    return extracted_values


def write_to_csv(extracted_values, output_path, column_header):
    # Check if the CSV already exists
    try:
        df_existing = pd.read_csv(output_path)
    except FileNotFoundError:
        df_existing = pd.DataFrame(columns=["Metric"])

    # Convert the dictionary of extracted values to a DataFrame
    df_new = pd.DataFrame(list(extracted_values.items()), columns=["Metric", column_header])

    # Merge the existing and new dataframes on the "Metric" column
    df_merged = pd.merge(df_existing, df_new, on="Metric", how="outer")

    # Export the merged DataFrame to CSV
    df_merged.to_csv(output_path, index=False)


# Define the possible combinations
datasets = ["MFPT", "XJTU"]
optimizers = ["Adam", "SGD"]
learning_rates = ["0.1", "0.01", "0.001", "0.0001"]

# Base directory
base_dir = "C:\\Users\\alrif\\Desktop\\Metrics and Visualizations 2\\MSCNNAM 7"

# Output CSV path
output_csv_path = "C:\\Users\\alrif\\Desktop\\Metrics and Visualizations 2\\MSCNNAM 7\\CMM.csv"

# Iterate over all combinations to extract and write values
for dataset in datasets:
    for optimizer in optimizers:
        for lr in learning_rates:
            folder_name = f"{dataset} ({optimizer}) ({lr})"
            file_path = f"{base_dir}/{folder_name}/MC.txt"

            try:
                values = extract_values_from_file(file_path)
                write_to_csv(values, output_csv_path, folder_name)
            except FileNotFoundError:
                # If the file doesn't exist for a particular combination, skip to the next
                pass
