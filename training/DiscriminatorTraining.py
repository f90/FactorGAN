import torch
import torch.nn

class DiscriminatorSetup(object):
    def __init__(self, name, D, optim, real_data, fake_data,
                 crop_real=lambda x:x, crop_fake=lambda x:x,
                 criterion=torch.nn.BCEWithLogitsLoss, real_label=1, fake_label=0):
        '''
        Disriminator model including optimiser, input sources, how inputs are cropped, and how the discriminator is trained
        :param name: Name of discriminator
        :param D: Discriminator model
        :param optim: Optimiser for D
        :param real_data: Real data iterator
        :param fake_data: Fake data iterator
        :param crop_real: Function that crops real inputs
        :param crop_fake: Function that crops fake inputs
        :param criterion: Discriminator training loss to use
        :param real_label: Real label for training loss
        :param fake_label: Fake label for training loss
        '''
        self.name = name
        self.D = D
        self.optim = optim
        self.real_data = real_data
        self.fake_data = fake_data

        self.real_label = real_label
        self.fake_label = fake_label

        self.crop_real = crop_real
        self.crop_fake = crop_fake

        self.criterion = criterion

class DependencyDiscriminatorSetup(object):
    def __init__(self, name, D, optim, data, shuffle_batch_func, crop_func=lambda x:x,
                 criterion=torch.nn.BCEWithLogitsLoss, real_label=1, fake_label=0):
        '''
        Dependency discriminator model including name, optimiser, input data source, how batches are shuffled, and training criterion
        :param name: Name of discriminator
        :param D: Discriminator model
        :param optim: Optimiser
        :param data: Joint data source ("real" data) with dependencies
        :param shuffle_batch_func: Function that shuffles a batch taken from data to get an independent version
        :param crop_func: Function that crops the input before feeding it to the discriminator
        :param criterion: Training loss
        :param real_label: Real label
        :param fake_label: Fake label (shuffled/independent batch)
        '''
        self.name = name
        self.D = D
        self.optim = optim
        self.data = data
        self.real_label = real_label
        self.fake_label = fake_label
        self.crop_func = crop_func
        self.shuffle_batch_func = shuffle_batch_func
        self.criterion = criterion

class DependencyDiscriminatorPair(object):
    def __init__(self, real_disc, fake_disc:DependencyDiscriminatorSetup):
        '''
        Binds to dependency discriminators (for p and q) together, to compute a combined output
        :param real_disc: p-dependency discriminator. Can be None in case none is used
        :param fake_disc: q-dependency discriminator
        '''
        assert(real_disc == None or isinstance(real_disc, DependencyDiscriminatorSetup))
        self.real_disc = real_disc
        self.fake_disc = fake_disc

    def get_comb_disc(self):
        '''
        Computes combined output d_P(x) - d_Q(x) that represents the dependency-based part of the generator loss
        :return:
        '''
        if self.real_disc != None:
            return lambda x : self.real_disc.D(self.real_disc.crop_func(x)) - self.fake_disc.D(self.fake_disc.crop_func(x))
        else:
            return lambda x : - self.fake_disc.D(self.fake_disc.crop_func(x))

def get_dep_disc_output(disc_setup:DependencyDiscriminatorSetup, device, backward=False, zero_gradients=False):
    '''
    Computes output of a dependency discriminator (and optionally gradients) for another input batch, by using the batch
    itself as real, and a shuffled version as fake input
    :param disc_setup: Dependency discriminator object
    :param device: Device to use
    :param backward: Whether to compute gradients
    :param zero_gradients: Whether to zero gradients at the beginning
    :return: see get_disc_output_batch
    '''
    real_sample = disc_setup.data.__next__()
    real_sample = disc_setup.crop_func(real_sample)

    fake_sample = disc_setup.shuffle_batch_func(real_sample) # Get fake data by simply shuffling current batch

    return get_disc_output_batch(disc_setup.D, real_sample, fake_sample, disc_setup.real_label, disc_setup.fake_label, disc_setup.criterion, device, backward, zero_gradients)

def get_marginal_disc_output(disc_setup:DiscriminatorSetup, device, backward=False, zero_gradients=False):
    '''
    Computes output of a marginal discriminator
    :param disc_setup: Marginal discriminator object
    :param device: Device to use
    :param backward: Whether to compute gradients
    :param zero_gradients: Whether to zero gradients at the beginning
    :return: see get_disc_output_batch
    '''
    real_sample = disc_setup.real_data.__next__().to(device)
    real_sample = disc_setup.crop_real(real_sample)

    fake_sample = disc_setup.fake_data.__next__().to(device)
    fake_sample = disc_setup.crop_fake(fake_sample)

    return get_disc_output_batch(disc_setup.D, real_sample, fake_sample, disc_setup.real_label, disc_setup.fake_label, disc_setup.criterion, device, backward, zero_gradients)

def get_disc_output_batch(D, real_sample, fake_sample, real_label, fake_label, criterion, device, backward, zero_gradients):
    '''
    Compute loss, output and optionally gradients for a discriminator model with a given training loss
    :param D: Discriminator model
    :param real_sample: Batch of real samples
    :param fake_sample: Batch of fake samples
    :param real_label: Target label for real batch
    :param fake_label: Target label for fake batch
    :param criterion: Training loss
    :param device: Device to use
    :param backward: Whether to compute gradients
    :param zero_gradients: Whether to zero gradients at the beginning
    :return: Average of real and fake training loss, discriminator accuracy, discriminator outputs for real and fake
    '''
    # Never backpropagate through disc input in this function
    # Transfer inputs to correct device
    real_sample = real_sample.detach().to(device)
    fake_sample = fake_sample.detach().to(device)

    if zero_gradients:
        D.zero_grad()

    # Get real sample output
    real_batch_size = real_sample.size()[0]

    label = torch.full((real_batch_size,), real_label, device=device)
    real_output = D(real_sample)

    errD_real = criterion()(real_output, label)
    if backward:
        errD_real.backward()

    # Get fake sample output
    fake_batch_size = fake_sample.size()[0]
    label = torch.full((fake_batch_size,), fake_label, device=device)
    fake_output = D(fake_sample)

    errD_fake = criterion()(fake_output, label)
    if backward:
        errD_fake.backward()

    # Accuracy
    correct = 0.5 * (real_output > 0.0).sum().item() / real_batch_size + 0.5 * (fake_output < 0.0).sum().item() / fake_batch_size
    return 0.5*errD_real + 0.5*errD_fake, correct, real_output, fake_output