import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("..\Datasets\datasets.csv")

# Getting the counts of different ages
age_counts = df['age'].value_counts()

# Getting the counts of different genders
gender_counts = df['gender'].value_counts()

# Define the labels for the plots
age_labels = {1: 'Young Adult', 2: 'Middle-Aged Adult', 3: 'Old-Aged Adult'}
gender_labels = {1: 'Female', 2: 'Male', 3: 'Other'}

# Create a figure with two subplots
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

# Plot for age counts
axs[0].bar(age_labels.values(), [age_counts.get(key, 0) for key in age_labels.keys()], color='c')
axs[0].set_title('Distribution of Age Groups')
axs[0].set_xlabel('Age Group')
axs[0].set_ylabel('Count')
axs[0].set_xticklabels(age_labels.values(), rotation=45, ha='right')

# Plot for gender counts
axs[1].bar(gender_labels.values(), [gender_counts.get(key, 0) for key in gender_labels.keys()], color='y')
axs[1].set_title('Distribution of Genders')
axs[1].set_xlabel('Gender')
axs[1].set_ylabel('Count')
axs[1].set_xticklabels(gender_labels.values(), rotation=45, ha='right')

# Adjust layout to prevent clipping of tick-labels
plt.tight_layout()
plt.show()
