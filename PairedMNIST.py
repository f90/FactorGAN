import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
import os
from torch.utils.data import DataLoader
import training.TrainingOptions
import training.AdversarialTraining
import Utils
from datasets.GeneratorInputDataset import GeneratorInputDataset
from datasets.InfiniteDataSampler import InfiniteDataSampler
from datasets.TransformDataSampler import TransformDataSampler
from datasets.DoubleMNISTDataset import DoubleMNISTDataset
from eval import Visualisation, FID
from training import MNIST
from training.DiscriminatorTraining import DiscriminatorSetup, DependencyDiscriminatorSetup, DependencyDiscriminatorPair
from models.discriminators.FCDiscriminator import FCDiscriminator
from models.generators.FCGenerator import FCGenerator
from datasets.CropDataset import CropDataset
import numpy as np

def set_paths(opt):
    # PATHS
    opt.experiment_path = os.path.join(opt.out_path, "PairedMNIST", opt.experiment_name)
    opt.gen_path = os.path.join(opt.experiment_path, "gen")
    opt.log_path = os.path.join(opt.experiment_path, "logs")
    Utils.make_dirs([opt.experiment_path, opt.gen_path, opt.log_path])

def predict_digits_batch(classifier, two_digit_input):
    '''
    Takes MNIST classifier and paired-MNIST sample and gives digit label probabilities for both
    :param classifier: MNIST classifier model
    :param two_digit_input: Paired MNIST sample
    :return: 20-dim. vector containing 2*10 digit label probabilities for upper and lower digit
    '''
    if len(two_digit_input.shape) == 2: # B, X
        two_digit_input = two_digit_input.view(-1, 1, 56, 28)
    elif len(two_digit_input.shape) == 3:  # B, H, W
        two_digit_input = two_digit_input.unsqueeze(1)

    upper_digit = two_digit_input[:, :, :28, :]
    lower_digit = two_digit_input[:, :, 28:, :]

    probs = torch.cat([classifier(upper_digit), classifier(lower_digit)], dim=1)
    return probs

def get_class_prob_matrix(G, G_inputs, classifier, num_samples, device):
    '''
    Build matrix of digit label combination frequencies (10x10)
    :param G: Generator model
    :param G_inputs: Input data sampler for generator (noise)
    :param classifier: MNIST classifier model
    :param num_samples: Number of samples to draw from generator to estimate c_Q
    :param device: Device to use
    :return: Normalised frequency of digit combination occurrences (10x10 matrix)
    '''
    it = 0
    joint_class_probs = np.zeros([10, 10])
    while(True):
        input_batch = next(G_inputs)
        # Get generator samples
        input_batch = [item.to(device) for item in input_batch]
        gen_batch = G(input_batch)
        # Feed through classifier
        digit_preds = predict_digits_batch(classifier, gen_batch)
        for pred in digit_preds:
            upper_pred = torch.argmax(pred[:10])
            lower_pred = torch.argmax(pred[10:])
            joint_class_probs[upper_pred, lower_pred] += 1

            it += 1
            if it >= num_samples:
                return joint_class_probs / np.sum(joint_class_probs)

def train(opt):
    print("Using " + str(opt.num_joint_samples) + " joint samples!")
    Utils.set_seeds(opt)
    device = Utils.get_device(opt.cuda)

    # DATA
    MNIST_dim = 784
    dataset = datasets.MNIST('datasets', train=True, download=True)

    # Create partitions of stacked MNIST
    dataset_joint = DoubleMNISTDataset(dataset, range(opt.num_joint_samples),same_digit_prob=opt.mnist_same_digit_prob)
    train_joint = InfiniteDataSampler(DataLoader(dataset_joint, num_workers=int(opt.workers), batch_size=opt.batchSize, shuffle=True))
    if opt.factorGAN == 1:
        # For marginals, take full dataset and crop it
        full_dataset = DoubleMNISTDataset(dataset, None, same_digit_prob=opt.mnist_same_digit_prob)
        train_x1 = InfiniteDataSampler(DataLoader(CropDataset(full_dataset, lambda x : x[:MNIST_dim]), num_workers=int(opt.workers), batch_size=opt.batchSize, shuffle=True))
        train_x2 = InfiniteDataSampler(DataLoader(CropDataset(full_dataset, lambda x : x[MNIST_dim:]), num_workers=int(opt.workers), batch_size=opt.batchSize, shuffle=True))

    # SETUP GENERATOR MODEL
    G = FCGenerator(opt, 2*MNIST_dim).to(device)
    G.train()
    G_noise = torch.distributions.uniform.Uniform(torch.Tensor([-1] * opt.nz), torch.Tensor([1] * opt.nz))
    G_opt = Utils.create_optim(G.parameters(), opt)

    # Prepare data sources that are a combination of real data and generator network, or purely from the generator network
    G_input_data = DataLoader(GeneratorInputDataset(None, G_noise), num_workers=int(opt.workers),
                              batch_size=opt.batchSize, shuffle=True)
    G_inputs = InfiniteDataSampler(G_input_data)
    G_outputs = TransformDataSampler(InfiniteDataSampler(G_input_data), G, device)

    # SETUP DISCRIMINATOR(S)
    if opt.factorGAN == 1:
        # Setup disc networks
        D1 = FCDiscriminator(MNIST_dim).to(device)
        D2 = FCDiscriminator(MNIST_dim).to(device)
        # If our dep discriminators are only defined on classifier probabilites, integrate classification into discriminator network as first step
        if opt.use_real_dep_disc == 1:
            DP = FCDiscriminator(2 * MNIST_dim,spectral_norm=(opt.lipschitz_p == 1)).to(device)
        else:
            DP = lambda x : 0

        DQ = FCDiscriminator(2 * MNIST_dim).to(device)

        # Prepare discriminators for training method
        # Marginal discriminators
        D1_setup = DiscriminatorSetup("D1", D1, Utils.create_optim(D1.parameters(), opt),
                                      train_x1, G_outputs, crop_fake=lambda x: x[:, :MNIST_dim])
        D2_setup = DiscriminatorSetup("D2", D2, Utils.create_optim(D2.parameters(), opt),
                                      train_x2, G_outputs, crop_fake=lambda x: x[:, MNIST_dim:])
        D_setups = [D1_setup, D2_setup]

        # Dependency discriminators
        shuffle_batch_func = lambda x: Utils.shuffle_batch_dims(x, marginal_index=MNIST_dim)

        if opt.use_real_dep_disc:
            DP_setup = DependencyDiscriminatorSetup("DP", DP, Utils.create_optim(DP.parameters(), opt), train_joint, shuffle_batch_func)
        else:
            DP_setup = None
        DQ_setup = DependencyDiscriminatorSetup("DQ", DQ, Utils.create_optim(DQ.parameters(), opt), G_outputs, shuffle_batch_func)
        D_dep_setups = [DependencyDiscriminatorPair(DP_setup, DQ_setup)]
    else:
        D = FCDiscriminator(2*MNIST_dim).to(device)
        D_setups = [DiscriminatorSetup("D", D, Utils.create_optim(D.parameters(), opt), train_joint, G_outputs)]
        D_dep_setups = []

    # RUN TRAINING
    training.AdversarialTraining.train(opt, G, G_inputs, G_opt, D_setups, D_dep_setups, device, opt.log_path)
    torch.save(G.state_dict(), os.path.join(opt.out_path, "G"))

def eval(opt):
    print("EVALUATING MNIST MODEL...")
    MNIST_dim = 784
    device = Utils.get_device(opt.cuda)

    # Train and save a digit classification model, needed for factorGAN variants and evaluation
    classifier = MNIST.main(opt)
    classifier.to(device)
    classifier.eval()

    # SETUP GENERATOR MODEL
    G = FCGenerator(opt, 2 * MNIST_dim).to(device)
    G_noise = torch.distributions.uniform.Uniform(torch.Tensor([-1] * opt.nz), torch.Tensor([1] * opt.nz))
    # Prepare data sources that are a combination of real data and generator network, or purely from the generator network
    G_input_data = DataLoader(GeneratorInputDataset(None, G_noise), num_workers=int(opt.workers),
                              batch_size=opt.batchSize, shuffle=True)
    G_inputs = InfiniteDataSampler(G_input_data)

    G.load_state_dict(torch.load(os.path.join(opt.experiment_path, opt.eval_model)))
    G.eval()

    # EVALUATE: Save images to eyeball them + FID for marginals + Class probability correlations
    writer = SummaryWriter(opt.log_path)

    test_mnist = datasets.MNIST('datasets', train=False, download=True)
    test_dataset = DoubleMNISTDataset(test_mnist, None, same_digit_prob=opt.mnist_same_digit_prob)
    test_dataset_loader = DataLoader(test_dataset, num_workers=int(opt.workers), batch_size=opt.batchSize, shuffle=True)
    transform_func = lambda x: x.view(-1, 1, 56, 28)
    Visualisation.generate_images(G, G_inputs, opt.gen_path, len(test_dataset), device, transform_func)

    crop_upper = lambda x: x[:, :, :28, :]
    crop_lower = lambda x: x[:, :, 28:, :]
    fid_upper = FID.evaluate_MNIST(opt, classifier, test_dataset_loader, opt.gen_path, device,crop_real=crop_upper,crop_fake=crop_upper)
    fid_lower = FID.evaluate_MNIST(opt, classifier, test_dataset_loader, opt.gen_path, device,crop_real=crop_lower,crop_fake=crop_lower)
    print("FID Upper Digit: " + str(fid_upper))
    print("FID Lower Digit: " + str(fid_lower))
    writer.add_scalar("FID_lower", fid_lower)
    writer.add_scalar("FID_upper", fid_upper)

    # ESTIMATE QUALITY OF DEPENDENCY MODELLING
    # cp(...) = cq(...) ideally for all inputs on the test set if dependencies are perfectly modelled. So compute average of that value and take difference to 1
    # Get joint distribution of real class indices in the data
    test_dataset = DoubleMNISTDataset(test_mnist, None,
                                      same_digit_prob=opt.mnist_same_digit_prob, deterministic=True, return_labels=True)
    test_it = DataLoader(test_dataset)
    real_class_probs = np.zeros((10, 10))
    for sample in test_it:
        _, d1, d2 = sample
        real_class_probs[d1, d2] += 1
    real_class_probs /= np.sum(real_class_probs)

    # Compute marginal distribution of real class indices from joint one
    real_class_probs_upper = np.sum(real_class_probs, axis=1)  # a
    real_class_probs_lower = np.sum(real_class_probs, axis=0)  # b
    real_class_probs_marginal = real_class_probs_upper * np.reshape(real_class_probs_lower, [-1, 1])

    # Get joint distribution of class indices on generated data (using classifier predictions)
    fake_class_probs = get_class_prob_matrix(G, G_inputs, classifier, len(test_dataset), device)
    # Compute marginal distribution of class indices on generated data
    fake_class_probs_upper = np.sum(fake_class_probs, axis=1)
    fake_class_probs_lower = np.sum(fake_class_probs, axis=0)
    fake_class_probs_marginal = fake_class_probs_upper * np.reshape(fake_class_probs_lower, [-1, 1])

    # Compute average of |cp(...) - cq(...)|
    cp = np.divide(real_class_probs, real_class_probs_marginal + 0.001)
    cq = np.divide(fake_class_probs, fake_class_probs_marginal + 0.001)

    diff_metric = np.mean(np.abs(cp - cq))

    print("Dependency cp/cq diff metric: " + str(diff_metric))
    writer.add_scalar("Diff-Dep", diff_metric)

    return fid_upper, fid_lower

def get_opt():
    # COLLECT ALL CMD ARGUMENTS
    parser = training.TrainingOptions.get_parser()

    parser.add_argument('--mnist_same_digit_prob', type=float, default=0.4,
                        help="Probability of same digits occuring together. 0.1 means indpendently put together, 1.0 means always same digits, 0.0 never same digits")
    parser.add_argument('--num_joint_samples', type=int, default=50,
                        help="Number of joint observations available for training normal gan/dependency discriminators")

    opt = parser.parse_args()
    # Set generator to sigmoid output
    opt.generator_activation = "sigmoid"
    print(opt)
    return opt

if __name__ == "__main__":
    opt = get_opt()

    set_paths(opt)

    if not opt.eval:
        train(opt)
    eval(opt)