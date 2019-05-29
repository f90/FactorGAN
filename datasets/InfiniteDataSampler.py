class InfiniteDataSampler(object):
    def __init__(self, data_loader):
        '''
        Can be used to infinitely loop over a dataset
        :param data_loader: Data loader object for a dataset
        '''
        self.data_loader = data_loader

        if not hasattr(data_loader, "__next__"):
            self.data_iter = iter(data_loader)
        else:
            self.data_iter = data_loader

    def next(self):
        return self.__next__()

    def __next__(self):
        try:
            data = next(self.data_iter)
        except StopIteration:
            # StopIteration is thrown if dataset ends
            # reinitialize data loader
            self.data_iter = iter(self.data_loader)
            data = next(self.data_iter)

        return data