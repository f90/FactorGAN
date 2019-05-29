import torch
import numpy as np

def lsgan_loss(model, real_input, fake_input, device):
    '''
    Compute LS-GAN loss function
    :param model: LS Discriminator model
    :param real_input: Real input data
    :param fake_input: Fake input data
    :param device: Device to use
    :return: LS loss
    '''
    real_output = model(real_input.to(device))
    fake_output = model(fake_input.to(device))

    a = 0.0
    b = 1.0

    return ((real_output - b)**2).mean() + ((fake_output - a)**2).mean()

def train_lsgan(model, real_data_loader, fake_data_loader, device):
    '''
    Trains LS discriminator model on real and fake data
    :param model: LS discriminator model
    :param real_data_loader: Real training data
    :param fake_data_loader: Fake training data
    :param device: Device to use
    '''
    model.train()
    NUM_EPOCHS = 40
    LR = 1e-4

    optim = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(NUM_EPOCHS):
        print("Epoch " + str(epoch))
        for real_batch in real_data_loader:
            fake_batch = next(fake_data_loader)

            optim.zero_grad()
            loss = lsgan_loss(model, real_batch, fake_batch, device)
            print(loss.item())
            loss.backward()
            optim.step()

    print("Finished training LSGAN Disc")

def lsgan_test_loss(model, real_data_loader, fake_data_loader, device):
    '''
    Obtains LS distance for pre-trained LS discriminator on test set
    :param model: Pre-trained LS discriminator
    :param real_data_loader: Real test data
    :param fake_data_loader: Fake test data
    :param device: Device to use
    :return: LS distance
    '''
    model.eval()

    batch_losses = list()
    batch_weights = list()
    for real_batch in real_data_loader:
        fake_batch = next(fake_data_loader)

        batch_losses.append(lsgan_loss(model, real_batch, fake_batch, device).item())
        batch_weights.append(real_batch.shape[0])

    total_loss = np.sum(np.array(batch_losses) * np.array(batch_weights)) / np.sum(np.array(batch_weights))
    print("LS: " + str(total_loss))
    return total_loss

def compute_ls_metric(classifier_factory, real_train, real_test, generator_sampler, repetitions, device):
    '''
    Computes LS metric for a generator model for evaluation
    :param classifier_factory: Function yielding a newly initialized (different each time!) LS classifier when called
    :param real_train: Real "test-train" dataset for training the LS discriminator
    :param real_test: Real "test-test" dataset for measuring LS distance after training LS discriminator
    :param generator_sampler: Generator output dataset
    :param repetitions: Number of times the classifier should be trained
    :param device: Device to use
    :return: List of LS distance metrics obtained for each training run (length "repetitions")
    '''
    losses = list()
    for _ in range(repetitions):
        classifier = classifier_factory()
        train_lsgan(classifier, real_train, generator_sampler, device)
        losses.append(lsgan_test_loss(classifier, real_test, generator_sampler, device))
        del classifier
    return losses