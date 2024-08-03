import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from read_datasets import gather_image_data


def plot_class_distribution(df):
    class_distribution = df['emotion'].value_counts()
    plt.figure(figsize=(10, 8))
    bars = class_distribution.plot(kind='bar')
    plt.title('Class Distribution')
    plt.xlabel('Emotion')
    plt.ylabel('Number of images')
    plt.xticks(rotation=45)

    # Add the exact number on each bar
    for bar in bars.patches:
        bars.annotate(format(bar.get_height(), '.0f'),
                      (bar.get_x() + bar.get_width() / 2,
                       bar.get_height()), ha='center', va='center',
                      size=10, xytext=(0, 8),
                      textcoords='offset points')
    plt.show()


def plot_sample_images(df):
    emotions = df['emotion'].unique()
    for emotion in emotions:
        fig, axs = plt.subplots(5, 5, figsize=(12, 12))
        fig.suptitle(f'Sample Images for {emotion}', fontsize=16)
        sample_df = df[df['emotion'] == emotion].sample(n=25, random_state=42)
        for idx, row in enumerate(sample_df.itertuples()):
            img = Image.open(row.image_path).resize((640, 640))
            ax = axs[idx // 5, idx % 5]
            ax.imshow(img)
            ax.axis('off')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


def plot_pixel_intensity_distribution(df):
    emotions = df['emotion'].unique()
    for emotion in emotions:
        plt.figure(figsize=(10, 8))
        # Initialize a flag to add legend labels only once
        added_legend = False
        sample_df = df[df['emotion'] == emotion].sample(n=25, random_state=42)
        for _, row in sample_df.iterrows():
            img = Image.open(row.image_path).resize((640, 640))  # Resize the image to 640x640 pixels
            pixels = np.array(img)
            if len(pixels.shape) == 3:  # Color image
                for color, channel in zip(['Red', 'Green', 'Blue'], range(3)):
                    intensity_distribution = pixels[:, :, channel].flatten()
                    if not added_legend:
                        plt.hist(intensity_distribution, bins=256, alpha=0.5, label=f'{color} Channel',
                                 color=color.lower())
                    else:
                        plt.hist(intensity_distribution, bins=256, alpha=0.5, color=color.lower())
            else:  # Grayscale image
                intensity_distribution = pixels.flatten()
                if not added_legend:
                    plt.hist(intensity_distribution, bins=256, alpha=0.5, label='Intensity')
                else:
                    plt.hist(intensity_distribution, bins=256, alpha=0.5)
            added_legend = True

        plt.title(f'Pixel Intensity Distribution for {emotion}')
        plt.legend()
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.show()


if __name__ == '__main__':
    # Load the dataset
    root_dir = '../Datasets'
    df = gather_image_data(root_dir)

    # Visualize class distribution
    plot_class_distribution(df)

    # Visualize sample images for each class
    plot_sample_images(df)

    # Visualize pixel intensity distribution for sample images
    plot_pixel_intensity_distribution(df)
