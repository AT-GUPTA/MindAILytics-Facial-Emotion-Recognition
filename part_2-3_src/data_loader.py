import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import numpy as np
import torch
import matplotlib as plt
from pathlib import Path
from sklearn.model_selection import train_test_split


# Custom Dataset
class FacesDataset(Dataset):
    def __init__(self, df, transform=None, secondary_transform=None):
        # List of img paths
        self.img_files = df['image_path']
        # List of equivalent emotion for each img
        self.labels = df['emotion']
        # List of equivalent age for each img
        self.group_ages = df['age']
        # List of equivalent gender for each img
        self.genders = df['gender']
        # Transformation to be applied to the images
        self.transform = transform
        # Transformation to be applied to the images for females
        self.secondary_transform = secondary_transform

    # Returns the number of images
    def __len__(self) -> int:
        return len(self.img_files)

    # Returns the img, emotion assigned, and name at the specified index
    def __getitem__(self, index):
        img_name = self.img_files[index]
        img = Image.open(img_name)
        # grayscale the img
        img = ImageOps.grayscale(img)
        label = self.labels[index]
        # check if a transformation is required
        gender = self.genders[index]
        if self.transform:
            img = self.transform(img)
        if gender == 1:
            if self.secondary_transform:
                img = self.secondary_transform(img)
        p = Path(img_name)
        # grab just the name of the img
        img_name = p.stem
        age = self.group_ages[index]
        gender = self.genders[index]
        return img, label, img_name, age, gender

    # Used to calculate the mean and std of the dataset to normalize it
    def mean_std_data(self):
        mean = 0
        std = 0
        for index in range(len(self.img_files)):
            img, _, _ = self.__getitem__(index)
            mean += torch.mean(img)
            std += torch.mean(img ** 2)

        mean = mean / len(self.img_files)
        std = (std / len(self.img_files) - mean ** 2) ** 0.5
        return mean, std

    # Helper method to display a sample of images
    def __showSampleImgs__(self):
        random_ix = np.random.randint(1, self.__len__(), size=4)
        fig = plt.figure(figsize=(6, 6))
        i = 1
        for idx in random_ix:
            ax = fig.add_subplot(2, 2, i)
            img = Image.open(self.img_files[idx])
            label = self.labels[idx]
            plt.imshow(img)
            plt.title(label, size=12)
            i += 1
        plt.tight_layout()
        plt.axis('off')
        plt.show()


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
                            if emotion == 'focused':
                                emotion = 0
                            if emotion == 'happy':
                                emotion = 1
                            if emotion == 'neutral':
                                emotion = 2
                            if emotion == 'surprised':
                                emotion = 3

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


def balanceDataFrame(df) -> pd.DataFrame:
    df_trimmed = df
    # Returns the minimum of rows grouped by emotions
    min_count = df_trimmed.groupby('emotion').count().min()[0]

    dfs = []
    # Loop through each emotion group
    for _, emotion in df_trimmed.groupby('emotion'):
        # Get the length of the emotion group
        size = len(emotion.squeeze())
        # If the length is greater than the minimum, drop the excess
        if size > min_count:
            # Get Random indices between the min and count of rows of the emotion
            drop_indices = np.random.choice(emotion.index, size=len(emotion) - min_count, replace=False)
            # Drop the rows
            emotion = emotion.drop(drop_indices)
        dfs.append(emotion)
    # Concatenate the dataframes
    df_balanced = pd.concat(dfs)
    # Shuffle the rows
    df_balanced = df_balanced.sample(frac=1).reset_index(drop=True)
    return df_balanced


def split_dataset(df, train_per=0.7, validation_per=0.15, random_state=42):
    train_dfs = []
    validation_dfs = []
    test_dfs = []

    validation_split_per = validation_per / (1-train_per)

    for _, emotion in df.groupby('emotion'):
        train_emotion, remainder_emotion = train_test_split(emotion, train_size=train_per, random_state=random_state)
        validation_emotion, test_emotion = train_test_split(remainder_emotion, train_size=validation_split_per,
                                                            random_state=random_state)

        train_dfs.append(train_emotion)
        validation_dfs.append(validation_emotion)
        test_dfs.append(test_emotion)

    train_df = pd.concat(train_dfs)
    validation_df = pd.concat(validation_dfs)
    test_df = pd.concat(test_dfs)

    train_df.reset_index(drop=True, inplace=True)
    validation_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    return train_df, validation_df, test_df


def add_bias(df, field, percentage_data_to_drop, cat_to_drop):
    # gender to create bias for
    field_df = df[df[f'{field}'] == cat_to_drop]

    # total number of entries to drop
    total_to_drop = int(len(field_df) * (percentage_data_to_drop / 100))
    counts_per_class = field_df['emotion'].value_counts()
    drop_per_class = (counts_per_class / counts_per_class.sum()) * total_to_drop
    drop_per_class = drop_per_class.apply(np.floor).astype(int)

    while drop_per_class.sum() < total_to_drop:
        drop_per_class[drop_per_class.index[3]] += 1

    # drop
    indices_to_drop = []
    for emotion, count_to_drop in drop_per_class.items():
        indices_of_class = field_df[field_df['emotion'] == emotion].index
        indices_to_drop.extend(np.random.choice(indices_of_class, count_to_drop, replace=False))

    df = df.drop(indices_to_drop)

    # shuffling the DataFrame to mix the records well
    df = df.sample(frac=1).reset_index(drop=True)

    return df


def dataset_details(df, field):
    details = df.groupby([f'{field}', 'emotion']).size().reset_index(name='count')

    details_pivot = details.pivot(index=f'{field}', columns='emotion', values='count')

    details_pivot = details_pivot.fillna(0).astype(int)

    return details_pivot
