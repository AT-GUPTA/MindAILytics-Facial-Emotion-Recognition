import os
import pandas as pd
from PIL import Image


def gather_image_data(root_dir='../Datasets'):

    data = []

    # Loop through each dataset directory
    for dataset_name in os.listdir(root_dir):
        dataset_path = os.path.join(root_dir, dataset_name)

        if os.path.isdir(dataset_path):
            # Loop through each emotion directory in the dataset
            for emotion in os.listdir(dataset_path):
                emotion_path = os.path.join(dataset_path, emotion)

                if os.path.isdir(emotion_path):
                    # Loop through each image in the emotion directory
                    for image_name in os.listdir(emotion_path):
                        image_path = os.path.join(emotion_path, image_name)

                        if os.path.isfile(image_path):
                            with Image.open(image_path) as img:
                                width, height = img.size
                            file_size = os.path.getsize(image_path)

                            # Append the data to the list
                            data.append({
                                'image_path': image_path,
                                'emotion': emotion,
                                'dataset_name': dataset_name,
                                'width': width,
                                'height': height,
                                'file_size': file_size
                            })

    # Convert to a pandas DataFrame
    df = pd.DataFrame(data)
    return df


if __name__ == '__main__':

    root_dir = '../Datasets'
    df = gather_image_data(root_dir)

    print(df.head())
