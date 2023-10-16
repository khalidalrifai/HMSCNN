import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re


# ==========================
# Data Extraction Function
# ==========================

def pad_list(lst, target_length, pad_value=float('nan')):
    return lst + [pad_value] * (target_length - len(lst))

def extract_info_from_file(filename, epochs=500):
    """
    Extracts training and validation loss and accuracy from a log file.
    """
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            train_loss_match = re.search('Train-Loss: ([\d\.nan]+)', line)
            val_loss_match = re.search('Validation-Loss: ([\d\.nan]+)', line)
            train_acc_match = re.search('Train-Accuracy: ([\d\.nan]+)', line)
            val_acc_match = re.search('Validation-Accuracy: ([\d\.nan]+)', line)

            if train_loss_match:
                train_losses.append(float(train_loss_match.group(1)))
            if train_acc_match:
                train_accs.append(float(train_acc_match.group(1)))
            if val_loss_match:
                val_losses.append(float(val_loss_match.group(1)))
            if val_acc_match:
                val_accs.append(float(val_acc_match.group(1)))

    # Ensure consistent lengths
    train_losses = pad_list(train_losses, epochs)
    val_losses = pad_list(val_losses, epochs)
    train_accs = pad_list(train_accs, epochs)
    val_accs = pad_list(val_accs, epochs)

    return train_losses, val_losses, train_accs, val_accs


# ======================
# Load Data from Files
# ======================

model_name = 'MSCNN 7 Model'
datasets = ["MFPT", "XJTU"]
optimizers_lr = [
    ('Adam', 0.001), ('Adam', 0.0001), ('Adam', 0.01), ('Adam', 0.1),
    ('SGD', 0.1), ('SGD', 0.01), ('SGD', 0.001), ('SGD', 0.0001)
]

# Dictionaries to store extracted data
train_losses, val_losses, train_accs, val_accs = {}, {}, {}, {}

base_path = "/Output"

# Placeholder for the data
data = []

for dataset in datasets:
    for optimizer, lr in optimizers_lr:
        file_path = f"{base_path}\\{model_name}\\{dataset}_{optimizer}_{lr}.log"
        train_l, val_l, train_a, val_a = extract_info_from_file(file_path)

        # Debugging line
        # print(f"Lengths - train_l: {len(train_l)}, val_l: {len(val_l)}, train_a: {len(train_a)}, val_a: {len(val_a)}")

        key = f"{dataset}_{optimizer}_{lr}"
        train_losses[key], val_losses[key], train_accs[key], val_accs[key] = train_l, val_l, train_a, val_a

        for epoch in range(len(train_l)):
            data.append([dataset, optimizer, lr, epoch + 1, train_l[epoch], val_l[epoch], train_a[epoch], val_a[epoch]])


# Convert the data to a DataFrame and save to CSV
df = pd.DataFrame(data, columns=['Dataset', 'Optimizer', 'Learning Rate', 'Epoch', 'Train Loss', 'Validation Loss', 'Train Accuracy', 'Validation Accuracy'])
csv_path = "C:\\Users\\alrif\\Desktop\\ThesisProject\\Output\\model_name.csv"
df.to_csv(csv_path, index=False)
print(f"Data saved to {csv_path}")

# ====================
# Plotting Functions
# ====================

# Use Seaborn's whitegrid style for a cleaner look
sns.set_style("whitegrid")

# Define a color palette with 4 colors.
color_palette = sns.color_palette("tab10", 8)

font_name, title_font_size, font_size, line_width = "Palatino Linotype", 20, 16, 2.5


def boldify_labels(ax):
    """Set the axis labels and title font weight to bold."""
    ax.xaxis.label.set_weight('bold')
    ax.yaxis.label.set_weight('bold')
    ax.title.set_weight('bold')


def plot_aesthetics(ax, xlabel, ylabel, title, legend_loc='upper right'):
    ax.tick_params(axis="both", which="both", bottom=True, top=False, left=True, right=False, length=5,
                   labelsize=font_size)
    ax.set_xlabel(xlabel, fontname=font_name, fontsize=font_size, labelpad=10)
    ax.set_ylabel(ylabel, fontname=font_name, fontsize=font_size, labelpad=10)
    ax.set_title(title, fontname=font_name, fontsize=title_font_size)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname(font_name)
    ax.legend(prop={'family': font_name, 'size': font_size - 2}, loc=legend_loc, frameon=True, edgecolor='gray')
    boldify_labels(ax)


def plot_combined(metric="loss", dataset="MFPT"):
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    data_source = {
        "loss": (train_losses, val_losses),
        "accuracy": (train_accs, val_accs)
    }
    train_data, val_data = data_source[metric]
    legend_location = 'lower right' if metric == 'accuracy' else 'upper right'
    for idx, opt in enumerate(['Adam', 'SGD']):
        ax = axes[idx]
        for color_id, (optimizer, lr) in enumerate(optimizers_lr):
            if optimizer == opt:
                key = f"{dataset}_{optimizer}_{lr}"
                ax.plot(train_data[key], linestyle='-', color=color_palette[color_id],
                        label=f'Training {metric.capitalize()} with Learning Rate of {lr}')
                ax.plot(val_data[key], linestyle='--', color=color_palette[color_id],
                        label=f'Validation {metric.capitalize()} with Learning Rate of {lr}')
        ylabel = 'Loss' if metric == "loss" else 'Accuracy'
        plot_aesthetics(ax, 'Epoch', ylabel, f'{metric.capitalize()} Curves ({opt} Optimizer on {dataset} Dataset)',
                        legend_loc=legend_location)
        # Limit y-axis for loss
        if metric == "loss":
         ax.set_ylim(bottom=0, top=5)  # Set y-axis limits here
        # Adding borders
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(2)
            spine.set_edgecolor('black')
    plt.tight_layout()
    plt.show()


def plot_accuracy_difference():
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    for idx, dataset in enumerate(["MFPT", "XJTU"]):
        ax = axes[idx]
        for color_id, (optimizer, lr) in enumerate(optimizers_lr):
            key = f"{dataset}_{optimizer}_{lr}"
            difference = [train - val for train, val in zip(train_accs[key], val_accs[key])]
            ax.plot(difference, linestyle='-', color=color_palette[color_id],
                    label=f'{optimizer} Optimizer with Learning Rate of {lr}')

        plot_aesthetics(ax, 'Epoch', 'Difference (Train - Validation)',
                        f'Training and Validation Accuracies Difference on {dataset} Dataset',
                        legend_loc='upper right')

        # Adding borders
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(2)
            spine.set_edgecolor('black')

    plt.tight_layout()
    plt.show()


# ================
# Execute Plotting
# ================

for metric in ["loss", "accuracy"]:
    for dataset in ["MFPT", "XJTU"]:
        plot_combined(metric, dataset)

plot_accuracy_difference()