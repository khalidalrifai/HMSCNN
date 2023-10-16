import os
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets.SequenceDatasets import dataset as sequence_dataset
from datasets.sequence_aug import *
from tqdm import tqdm

ROOT_PATH = "C:\\Users\\alrif\\Desktop\\ThesisProject\\XJTU\\"
SIGNAL_SIZE = 1024
LABEL_RANGES = [
    range(0, 5),
    range(5, 10),
    range(10, 15)
]


def get_files_for_datasetname(root, wc, datasetname, label_range):
    data, lab = [], []
    for idx in tqdm(range(len(datasetname))):
        files = os.listdir(os.path.join(root, wc, datasetname[idx]))
        for last_file_idx in [-4, -3, -2, -1]:
            path = os.path.join(root, wc, datasetname[idx], files[last_file_idx])
            data_part, lab_part = data_load(path, label=label_range[idx])
            data.extend(data_part)
            lab.extend(lab_part)
    return data, lab


def get_files(root):
    working_conditions = os.listdir(root)
    data, lab = [], []

    for wc, label_range in zip(working_conditions, LABEL_RANGES):
        datasetname = os.listdir(os.path.join(root, wc))
        data_part, lab_part = get_files_for_datasetname(root, wc, datasetname, label_range)
        data.extend(data_part)
        lab.extend(lab_part)

    return [data, lab]


def data_load(filename, label):
    fl = pd.read_csv(filename)
    fl = fl["Horizontal_vibration_signals"].values.reshape(-1, 1)
    data, lab = [], []
    start, end = 0, SIGNAL_SIZE
    while end <= len(fl):
        data.append(fl[start:end])
        lab.append(label)
        start += SIGNAL_SIZE
        end += SIGNAL_SIZE
    return data, lab


def data_transforms(dataset_type="train", normlize_type="-1-1"):
    if dataset_type == "train":
        transforms = [
            Reshape(),
            Normalize(type=normlize_type),
            AddGaussianNoise(sigma=0.01, prob=1.0),  # Here the default sigma and prob values are kept. Modify if needed.
            RandomJitter(sigma=0.005),  # Default sigma value is kept. Modify if needed.
            RandomStretch(sigma=0.3),  # Default sigma value is kept. Modify if needed.
            RandomTimeWarping(sigma=0.2),  # Default sigma value is kept. Modify if needed.
            RandomDropout(dropout_prob=0.05),  # Default dropout_prob value is kept. Modify if needed.
            RandomCrop(crop_len=20),  # Default crop_len value is kept. Modify if needed.
            RandomFlip(),
            RandomFill(fill_len=20, fill_type="mean"),  # Default values are kept. Modify if needed.
            RandomShift(shift_range=10),  # Default shift_range value is kept. Modify if needed.
            RandomFrequencyNoise(sigma=0.01),  # Default sigma value is kept. Modify if needed.
            Retype()
        ]
    else:
        transforms = [
            Reshape(),
            Normalize(type=normlize_type),
            Retype()
        ]

    return Compose(transforms)



class XJTU:
    num_classes = 15
    inputchannel = 1

    def __init__(self, data_dir, normlizetype):
        self.data_dir = data_dir
        self.normlizetype = normlizetype

    def data_prepare(self):
        list_data = get_files(self.data_dir)

        # Create the initial dataframe from the fetched data
        data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})

        # Step 1: Splitting data into training (70%) and a temporary set (30%)
        train_pd, temp_pd = train_test_split(data_pd, test_size=0.30, random_state=40, stratify=data_pd["label"])

        # Step 2: Splitting the temporary set into validation (20%) and testing (10%) sets
        val_pd, test_pd = train_test_split(temp_pd, test_size=1 / 3, random_state=40, stratify=temp_pd["label"])

        train_dataset = sequence_dataset(list_data=train_pd, transform=data_transforms('train', self.normlizetype))
        val_dataset = sequence_dataset(list_data=val_pd, transform=data_transforms('val', self.normlizetype))
        test_dataset = sequence_dataset(list_data=test_pd, transform=data_transforms('test', self.normlizetype))

        return train_dataset, val_dataset, test_dataset
