from torch.utils.data.dataset import Dataset

class CropDataset(Dataset):
    def __init__(self, dataset, crop_func):
        self.dataset = dataset
        self.crop_func = crop_func

    def __getitem__(self, index):
        sample = self.dataset[index]
        return self.crop_func(sample)

    def __len__(self):
        return len(self.dataset)