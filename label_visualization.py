import pandas as pd
import matplotlib.pyplot as plt


def bar_chart_sample_per_label_per_year(data, output_path = None):
    # Count the number of samples per label per year
    samples_per_label_per_year = data.groupby(['label', 'year']).size().reset_index(name='count')

    # Pivot the table to create a separate column for each label
    samples_per_label_per_year = samples_per_label_per_year.pivot(index='year', columns='label', values='count')

    # Plot a bar graph of the results
    ax = samples_per_label_per_year.plot(kind='bar', stacked=False, figsize=(8, 6))
    ax.set_xlabel('Year')
    ax.set_ylabel('Samples')
    ax.set_title('Samples per Label per Year')
    plt.tight_layout()
    
    if output_path != None:
        plt.savefig(output_path)
    plt.show()

def pie_chart_label_distribution(data, output_path = None):
    # Create a pie chart of the label distribution across years
    counts = data['label'].value_counts()
    ax = counts.plot(kind='pie', autopct='%1.1f%%')
    ax.set_title('Label Distribution Across Years')
    if output_path != None:
        plt.savefig(output_path)
    plt.show()



# Load the data
data = pd.read_csv("test_output_test1.csv")
bar_chart_sample_per_label_per_year(data, output_path = 'samples_per_label_per_year.png')
pie_chart_label_distribution(data, output_path = 'label_distribution_across_years.png')