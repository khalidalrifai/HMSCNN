from torch.utils.data import Dataset
from datasets.sequence_aug import *


class dataset(Dataset):

    def __init__(self, list_data, test=False, transform=None):
        self.test = test
        self.seq_data = list_data['data'].tolist()

        if not self.test:
            self.labels = list_data['label'].tolist()

        if transform is None:
            self.transforms = Compose([
                Reshape()
            ])
        else:
            self.transforms = transform

    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, item):
        seq = self.seq_data[item]
        seq = self.transforms(seq)

        if self.test:
            return seq, item
        else:
            label = self.labels[item]
            return seq, label

