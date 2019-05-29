from torch.utils.data.dataset import Dataset
import numpy as np

class GeneratorInputDataset(Dataset):
    def __init__(self, cond_dataset, noise_distribution):
        assert(cond_dataset != None or noise_distribution != None)
        self.cond_dataset = cond_dataset
        self.noise = noise_distribution

    def __getitem__(self, index):
        output = list()

        if self.cond_dataset != None:
            # Sample input for generator from other dataset
            output.append(self.cond_dataset[np.random.randint(0, len(self.cond_dataset))])

        if self.noise != None:
            # Sample noise from noise_distribution, if we are using noise in the conditional generator network
            output.append(self.noise.sample())

        # Get generator input
        return output

    def __len__(self):
        if self.cond_dataset != None:
            return len(self.cond_dataset)
        else:
            return 10000