import os
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from datasets.SequenceDatasets import dataset as sequence_dataset
from datasets.sequence_aug import *
from tqdm import tqdm

SIGNAL_SIZE = 1024
ROOT_PATH = "C:\\Users\\alrif\\Desktop\\ThesisProject\\MFPT\\"

BASELINE_LABEL = 0
OUTER_RACE_LABELS = list(range(1, 8))
INNER_RACE_LABELS = list(range(8, 15))


def get_files(root_path):
    """
    Generate the final training set and test set.
    root_path: The location of the data set
    """
    categories = sorted(os.listdir(root_path))

    data_roots = {
        'baseline': os.path.join(root_path, categories[0]),
        'outer_race': os.path.join(root_path, categories[2]),
        'inner_race': os.path.join(root_path, categories[3])
    }

    # Load baseline data
    baseline_files = os.listdir(data_roots['baseline'])
    baseline_path = os.path.join(data_roots['baseline'], baseline_files[2])

    data, labels = data_load(baseline_path, BASELINE_LABEL)

    # Load outer race fault conditions data
    outer_race_files = os.listdir(data_roots['outer_race'])
    for i, filename in tqdm(enumerate(outer_race_files), total=len(outer_race_files)):
        path = os.path.join(data_roots['outer_race'], filename)
        data_chunk, labels_chunk = data_load(path, OUTER_RACE_LABELS[i])
        data += data_chunk
        labels += labels_chunk

    # Load inner race fault conditions data
    inner_race_files = os.listdir(data_roots['inner_race'])
    for j, filename in tqdm(enumerate(inner_race_files), total=len(inner_race_files)):
        path = os.path.join(data_roots['inner_race'], filename)
        data_chunk, labels_chunk = data_load(path, INNER_RACE_LABELS[j])
        data += data_chunk
        labels += labels_chunk

    return [data, labels]


def data_load(filename, label):
    """
    Generate test data and training data.
    filename: Data location
    """
    mat_data = loadmat(filename)
    if label == BASELINE_LABEL:
        data_array = mat_data["bearing"][0][0][1]
    else:
        data_array = mat_data["bearing"][0][0][2]

    segments, segment_labels = [], []
    start, end = 0, SIGNAL_SIZE
    while end <= data_array.shape[0]:
        segments.append(data_array[start:end])
        segment_labels.append(label)
        start += SIGNAL_SIZE
        end += SIGNAL_SIZE

    return segments, segment_labels


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


class MFPT:
    num_classes = 15
    inputchannel = 1

    def __init__(self, data_dir, normalize_type):
        self.data_dir = data_dir
        self.normalize_type = normalize_type

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
