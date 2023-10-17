import os
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets.SequenceDatasets import dataset as sequence_dataset
from datasets.sequence_aug import *
from tqdm import tqdm

signal_size = 1024

# Three working conditions
WC1 = ["ib600_2.csv", "n600_3_2.csv", "ob600_2.csv", "tb600_2.csv"]
WC2 = ["ib800_2.csv", "n800_3_2.csv", "ob800_2.csv", "tb800_2.csv"]
WC3 = ["ib1000_2.csv", "n1000_3_2.csv", "ob1000_2.csv", "tb1000_2.csv"]

label1 = [i for i in range(0, 4)]
label2 = [i for i in range(4, 8)]
label3 = [i for i in range(8, 12)]


# generate Training Dataset and Testing Dataset
def get_files(root, test=False):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    '''

    data = []
    lab = []
    for i in tqdm(range(len(WC1))):
        path1 = os.path.join('/tmp', root, WC1[i])
        data1, lab1 = data_load(path1, label=label1[i])
        data += data1
        lab += lab1

    for j in tqdm(range(len(WC2))):
        path2 = os.path.join('/tmp', root, WC2[j])
        data2, lab2 = data_load(path2, label=label2[j])
        data += data2
        lab += lab2

    for k in tqdm(range(len(WC3))):
        path3 = os.path.join('/tmp', root, WC3[k])
        data3, lab3 = data_load(path3, label=label3[k])
        data += data3
        lab += lab3

    return [data, lab]


def data_load(filename, label):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    '''
    fl = np.loadtxt(filename)
    fl = fl.reshape(-1, 1)
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


class JNU(object):
    num_classes = 12
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
