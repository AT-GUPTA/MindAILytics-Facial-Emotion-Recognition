from read_datasets import gather_image_data


def analyze_image_dimensions(df):

    median_height = df['height'].median()
    median_width = df['width'].median()

    smallest = df.loc[(df['height'] * df['width']).idxmin()]
    largest = df.loc[(df['height'] * df['width']).idxmax()]

    print(f"Median Dimensions: {median_width} x {median_height}")
    print(f"Smallest Image: {smallest['width']} x {smallest['height']} (File size: {smallest['file_size']} bytes)")
    print(f"Largest Image: {largest['width']} x {largest['height']} (File size: {largest['file_size']} bytes)")

    # Distribution of images per height x width
    dimension_distribution = df.groupby(['width', 'height']).size().reset_index(name='count')
    print("\nDistribution of number of images per height x width bracket:")
    print(dimension_distribution.sort_values(by='count', ascending=False))
    return median_width


def recommend_resize_strategy(df,median_width):

    aspect_ratios = df['width'] / df['height']
    common_aspect_ratio = aspect_ratios.median()

    # Suggest a resize dimension based on the median aspect ratio and the median dimensions
    suggested_width = round(median_width)
    suggested_height = round(suggested_width / common_aspect_ratio)

    print(f"\nSuggested Resize Dimensions (to maintain aspect ratio): {suggested_width} x {suggested_height}")
    return suggested_width, suggested_height


if __name__ == '__main__':
    df = gather_image_data()

    wd = analyze_image_dimensions(df)
    recommend_resize_strategy(df, wd)
