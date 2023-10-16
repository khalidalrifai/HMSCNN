import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

# Set font and style parameters
font_name, title_font_size, font_size, line_width = "Palatino Linotype", 20, 16, 2.5

# Define the font properties for titles and general use
title_font_prop = FontProperties()
title_font_prop.set_family(font_name)
title_font_prop.set_size(title_font_size)

font_prop = FontProperties()
font_prop.set_family(font_name)
font_prop.set_size(font_size)

# Path to the data file
input_file = "/Output/Best.csv"

# Load data
data = pd.read_csv(input_file)

# Set the plotting style
sns.set_style("whitegrid")

# Columns to be plotted
loss_columns = ['Train Loss', 'Validation Loss']
precision_columns = ['Train Precision', 'Validation Precision']

def plot_box(data, columns, title_prefix):
    """Plot boxplots for given columns."""
    plt.figure(figsize=(20, 10))
    for idx, column in enumerate(columns, 1):
        plt.subplot(1, 2, idx)
        sns.boxplot(data=data, x='Model', y=column, hue='Dataset', linewidth=line_width)
        plt.title(f'Box Plot of {column}', fontproperties=title_font_prop)
        plt.ylabel(column, fontproperties=font_prop)
        plt.xlabel('Model', fontproperties=font_prop)
        plt.legend(title='Dataset', loc='upper right', prop=font_prop)
        plt.setp(plt.gca().get_xticklabels(), fontproperties=font_prop)
        plt.setp(plt.gca().get_yticklabels(), fontproperties=font_prop)
    plt.tight_layout()
    plt.show()

def plot_histogram(data, columns, title_prefix):
    """Plot separate histograms for each dataset in the given columns."""
    for column in columns:
        plt.figure(figsize=(20, 10))
        for idx, dataset in enumerate(data['Dataset'].unique(), 1):
            plt.subplot(1, 2, idx)
            for model in data['Model'].unique():
                subset = data[(data['Model'] == model) & (data['Dataset'] == dataset)]
                sns.histplot(subset[column], label=f'{model}', kde=True, bins=30, linewidth=line_width)
            plt.title(f'Histogram of {column} for {dataset}', fontproperties=title_font_prop)
            plt.ylabel('Frequency', fontproperties=font_prop)
            plt.xlabel(column, fontproperties=font_prop)
            plt.legend(prop=font_prop)
            plt.setp(plt.gca().get_xticklabels(), fontproperties=font_prop)
            plt.setp(plt.gca().get_yticklabels(), fontproperties=font_prop)
        plt.tight_layout()
        plt.show()

# Plotting
plot_box(data, loss_columns, "Loss")
plot_box(data, precision_columns, "Precision")
plot_histogram(data, loss_columns, "Loss")
plot_histogram(data, precision_columns, "Precision")
