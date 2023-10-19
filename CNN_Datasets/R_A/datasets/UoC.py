import os
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from datasets.SequenceDatasets import dataset as sequence_dataset
from datasets.sequence_aug import *
from tqdm import tqdm

signal_size = 1024

dataname = ["DataForClassification_Stage0.mat", "DataForClassification_TimeDomain.mat"]
# label
label = [0, 1, 2, 3, 4, 5, 6, 7,
         8]  # The data is labeled 0-8,they are {‘healthy’,‘missing’,‘crack’,‘spall’,‘chip5a’,‘chip4a’,‘chip3a’,‘chip2a’,‘chip1a’}


# generate Training Dataset and Testing Dataset
def get_files(root, test=False):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    '''
    path = os.path.join('/tmp', root, dataname[1])
    data = loadmat(path)
    data = data['AccTimeDomain']
    da = []
    lab = []

    start, end = 0, 104  # Number of samples per type=104
    i = 0
    while end <= data.shape[1]:
        data1 = data[:, start:end]
        data1 = data1.reshape(-1, 1)
        da1, lab1 = data_load(data1, label=label[i])
        da += da1
        lab += lab1
        start += 104
        end += 104
        i += 1
    return [da, lab]


def data_load(fl, label):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    '''
    data = []
    lab = []
    start, end = 0, signal_size
    while end <= fl.shape[0]:
        data.append(fl[start:end])
        lab.append(label)
        start += signal_size
        end += signal_size

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


class UoC(object):
    num_classes = 9
    inputchannel = 1

    def __init__(self, data_dir, normlizetype):
        self.data_dir = data_dir
        self.normlizetype = normlizetype

    def data_prepare(self, test=False):
        list_data = get_files(self.data_dir)

        # Create the initial dataframe from the fetched data
        data_df = pd.DataFrame({"data": list_data[0], "label": list_data[1]})

        # First split: 70% train, 30% temp
        train_df, temp_df = train_test_split(data_df, test_size=0.3, random_state=40, stratify=data_df["label"])

        # Second split: 2/3 validation (20% of original), 1/3 test (10% of original)
        val_df, test_df = train_test_split(temp_df, test_size=1 / 3, random_state=40, stratify=temp_df["label"])

        train_dataset = sequence_dataset(list_data=train_df, transform=data_transforms('train', self.normalize_type))
        val_dataset = sequence_dataset(list_data=val_df, transform=data_transforms('val', self.normalize_type))
        test_dataset = sequence_dataset(list_data=test_df, transform=data_transforms('test', self.normalize_type))

        return train_dataset, val_dataset, test_dataset
