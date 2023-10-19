import os
import pandas as pd
from itertools import islice
from sklearn.model_selection import train_test_split
from datasets.SequenceDatasets import dataset as sequence_dataset
from datasets.sequence_aug import *
from tqdm import tqdm

signal_size = 1024

# Data names of 5 bearing fault types under two working conditions
Bdata = ["ball_20_0.csv", "comb_20_0.csv", "health_20_0.csv", "inner_20_0.csv", "outer_20_0.csv", "ball_30_2.csv",
         "comb_30_2.csv", "health_30_2.csv", "inner_30_2.csv", "outer_30_2.csv"]
label1 = [i for i in range(0, 10)]
# Data names of 5 gear fault types under two working conditions
Gdata = ["Chipped_20_0.csv", "Health_20_0.csv", "Miss_20_0.csv", "Root_20_0.csv", "Surface_20_0.csv",
         "Chipped_30_2.csv", "Health_30_2.csv", "Miss_30_2.csv", "Root_30_2.csv", "Surface_30_2.csv"]
labe12 = [i for i in range(10, 20)]


# generate Training Dataset and Testing Dataset
def get_files(root, test=False):
    """
    This function is used to generate the final training set and test set.
    root:The location of the data set
    datasetname:List of  dataset
    """
    datasetname = os.listdir(os.path.join(root, os.listdir(root)[2]))  # 0:bearingset, 2:gearset
    root1 = os.path.join("/tmp", root, os.listdir(root)[2], datasetname[0])  # Path of bearingset
    root2 = os.path.join("/tmp", root, os.listdir(root)[2], datasetname[2])  # Path of gearset

    data = []
    lab = []
    for i in tqdm(range(len(Bdata))):
        path1 = os.path.join('/tmp', root1, Bdata[i])
        data1, lab1 = data_load(path1, dataname=Bdata[i], label=label1[i])
        data += data1
        lab += lab1

    for j in tqdm(range(len(Gdata))):
        path2 = os.path.join('/tmp', root2, Gdata[j])
        data2, lab2 = data_load(path2, dataname=Gdata[j], label=labe12[j])
        data += data2
        lab += lab2

    return [data, lab]


def data_load(filename, dataname, label):
    """
    This function is mainly used to generate test data and training data.
    filename:Data location
    """
    f = open(filename, "r", encoding='gb18030', errors='ignore')
    fl = []
    if dataname == "ball_20_0.csv":
        for line in islice(f, 16, None):  # Skip the first 16 lines
            line = line.rstrip()
            word = line.split(",", 8)  # Separated by commas
            fl.append(eval(word[1]))  # Take a vibration signal in the x direction as input
    else:
        for line in islice(f, 16, None):  # Skip the first 16 lines
            line = line.rstrip()
            word = line.split("\t", 8)  # Separated by \t
            fl.append(eval(word[1]))  # Take a vibration signal in the x direction as input
    fl = np.array(fl)
    fl = fl.reshape(-1, 1)
    data = []
    lab = []
    start, end = 0, signal_size
    while end <= fl.shape[0] / 10:
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


class SEU(object):
    num_classes = 20
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
