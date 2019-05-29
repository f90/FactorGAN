from torch.utils.data.dataset import Dataset
import torch
import numpy as np

class DoubleMNISTDataset(Dataset):
    def __init__(self, mnist_dataset, idx_range, deterministic=True, same_digit_prob=0.9, return_labels=False):
        if idx_range is None:
            idx_range = range(len(mnist_dataset.data))
        self.data = mnist_dataset.data[idx_range].clone().detach().float() / 255.0
        self.labels = mnist_dataset.targets[idx_range].clone().detach()

        self.idx_range = range(len(idx_range))

        self.same_digit_prob = same_digit_prob
        self.deterministic = deterministic

        self.idx_digits = list()
        self.idx_digits_proportion = list()
        for digit_num in range(10):
            indices = [i for i in self.idx_range if self.labels[i] == digit_num]
            self.idx_digits.append(indices)
            self.idx_digits_proportion.append(float(len(indices)))
        self.idx_digits_proportion /= np.sum(self.idx_digits_proportion)

        self.second_digits = np.full(len(self.idx_range), -1)

        self.return_labels = return_labels

    def __getitem__(self, index):
        first_digit, first_digit_label = self.data[index], self.labels[index]
        if self.deterministic and self.second_digits[index] != -1:
            second_digit_idx = self.second_digits[index]
        else:
            # Draw same digit with given higher probability
            same_digit = (np.random.rand() < self.same_digit_prob)

            if same_digit:
                # Draw from pool of same-digit numbers
                second_digit_label = first_digit_label
            else:
                # Draw from pool of different-digit numbers
                second_digit_label = np.random.choice(range(10), p=self.idx_digits_proportion)
            second_digit_idx = self.idx_digits[second_digit_label.item()][np.random.randint(0, len(self.idx_digits[second_digit_label]))]

            # Finally, save second digit choice if deterministic
            if self.deterministic:
                self.second_digits[index] = second_digit_idx

        sample = torch.cat([first_digit.view(-1), self.data[second_digit_idx].view(-1)])
        if self.return_labels:
            return (sample, first_digit_label, second_digit_label)
        else:
            return sample


    def __len__(self):
        return len(self.idx_range)

    def get_digits(self, digit_label):
        return sum([self.idx_digits[d] for d in range(10) if d != digit_label], [])