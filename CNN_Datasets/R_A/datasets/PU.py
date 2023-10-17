import os
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from datasets.SequenceDatasets import dataset as sequence_dataset
from datasets.sequence_aug import *
from tqdm import tqdm

signal_size = 1024

# 1 Undamaged (healthy) bearings(6X)
HBdata = ['K001', "K002", 'K003', 'K004', 'K005', 'K006']
label1 = [0, 1, 2, 3, 4, 5]  # The undamaged (healthy) bearings data is labeled 1-9
# 2 Artificially damaged bearings(12X)
ADBdata = ['KA01', 'KA03', 'KA05', 'KA06', 'KA07', 'KA08', 'KA09', 'KI01', 'KI03', 'KI05', 'KI07', 'KI08']
label2 = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]  # The artificially damaged bearings data is labeled 4-15
# 3 Bearings with real damages caused by accelerated lifetime tests(14x)
# RDBdata = ['KA04','KA15','KA16','KA22','KA30','KB23','KB24','KB27','KI04','KI14','KI16','KI17','KI18','KI21']
# label3=[18,19,20,21,22,23,24,25,26,27,28,29,30,31]  #The artificially damaged bearings data is labeled 16-29
RDBdata = ['KA04', 'KA15', 'KA16', 'KA22', 'KA30', 'KB23', 'KB24', 'KB27', 'KI14', 'KI16', 'KI17', 'KI18', 'KI21']
label3 = [i for i in range(13)]

# working condition
WC = ["N15_M07_F10", "N09_M07_F10", "N15_M01_F10", "N15_M07_F04"]
state = WC[0]  # WC[0] can be changed to different working states


# generate Training Dataset and Testing Dataset
def get_files(root, test=False):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    '''
    data = []
    lab = []

    for k in tqdm(range(len(RDBdata))):
        name3 = state + "_" + RDBdata[k] + "_1"
        path3 = os.path.join('/tmp', root, RDBdata[k], name3 + ".mat")
        data3, lab3 = data_load(path3, name=name3, label=label3[k])
        data += data3
        lab += lab3

    return [data, lab]


def data_load(filename, name, label):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    '''
    fl = loadmat(filename)[name]
    fl = fl[0][0][2][0][6][2]  # Take out the data
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
            AddGaussianNoise(sigma=0.01, prob=1.0),
            # Here the default sigma and prob values are kept. Modify if needed.
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


class PU(object):
    num_classes = 13
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
