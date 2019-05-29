class TransformDataSampler(object):
    def __init__(self, data_loader, transform, transform_device):
        # create dataloader-iterator
        self.data_loader = data_loader

        if not hasattr(data_loader, "__next__"):
            self.data_iter = iter(data_loader)
        else:
            self.data_iter = data_loader

        self.transform = transform
        self.device = transform_device

    def next(self):
        return self.__next__()

    def __next__(self):
        data = next(self.data_iter)

        # Put transform input to proper device first
        if self.device != None:
            if isinstance(data, list):
                data = [item.to(self.device) for item in data]
            else:
                data = data.to(self.device)

        # Transform batch of samples
        data = self.transform(data).detach()

        return data