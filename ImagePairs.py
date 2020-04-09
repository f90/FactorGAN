import csv
import torch
import os
from torch.utils.data import DataLoader, Subset

import training.TrainingOptions
import training.AdversarialTraining
import Utils
from datasets.GeneratorInputDataset import GeneratorInputDataset
from datasets.InfiniteDataSampler import InfiniteDataSampler
from datasets.TransformDataSampler import TransformDataSampler
from datasets.image2image import get_aligned_dataset
from eval import LS
from eval.Visualisation import generate_images
from models.generators.ConvGenerator import ConvGenerator
from training.DiscriminatorTraining import DiscriminatorSetup, DependencyDiscriminatorSetup, DependencyDiscriminatorPair
from models.discriminators.ConvDiscriminator import ConvDiscriminator
from datasets.CropDataset import CropDataset

def set_paths(opt):
    # Set up paths and create folders
    opt.experiment_path = os.path.join(opt.out_path, "ImagePairs", opt.dataset, opt.experiment_name)
    opt.gen_path = os.path.join(opt.experiment_path, "gen")
    opt.log_path = os.path.join(opt.experiment_path, "logs")
    Utils.make_dirs([opt.experiment_path, opt.gen_path, opt.log_path])

def train(opt):
    Utils.set_seeds(opt)
    device = Utils.get_device(opt.cuda)
    set_paths(opt)

    # DATA
    dataset = get_aligned_dataset(opt, "train")
    nc = dataset.A_nc + dataset.B_nc

    # Warning if desired number of joint samples is larger than dataset, in that case, use whole dataset as paired
    if opt.num_joint_samples > len(dataset):
        print("WARNING: Cannot train with " + str(opt.num_joint_samples) + " samples, dataset has only size of " + str(len(dataset))+ ". Using full dataset!")
        opt.num_joint_samples = len(dataset)

    # Joint samples
    dataset_train = Subset(dataset, range(opt.num_joint_samples))
    train_joint = InfiniteDataSampler(
        DataLoader(dataset_train, num_workers=int(opt.workers), batch_size=opt.batchSize, shuffle=True, drop_last=True))

    if opt.factorGAN == 1:
        # For marginals, take full dataset and crop
        train_a = InfiniteDataSampler(DataLoader(CropDataset(dataset, lambda x : x[0:dataset.A_nc, :, :]),
                                                 num_workers=int(opt.workers), batch_size=opt.batchSize, shuffle=True))
        train_b = InfiniteDataSampler(DataLoader(CropDataset(dataset, lambda x : x[dataset.A_nc:, :, :]),
                                                 num_workers=int(opt.workers), batch_size=opt.batchSize, shuffle=True))

    # SETUP GENERATOR MODEL
    G = ConvGenerator(opt, opt.generator_channels, opt.loadSize, nc).to(device)
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
        D1 = ConvDiscriminator(opt.loadSize, opt.loadSize, dataset.A_nc, opt.disc_channels).to(device)
        D2 = ConvDiscriminator(opt.loadSize, opt.loadSize, dataset.B_nc, opt.disc_channels).to(device)
        # If our dep discriminators are only defined on classifier probabilites, integrate classification into discriminator network as first step
        if opt.use_real_dep_disc == 1:
            DP = ConvDiscriminator(opt.loadSize, opt.loadSize, nc, opt.disc_channels, spectral_norm=(opt.lipschitz_p == 1)).to(device)
        else:
            DP = lambda x : 0

        DQ = ConvDiscriminator(opt.loadSize, opt.loadSize, nc, opt.disc_channels).to(device)
        print(sum(p.numel() for p in D1.parameters()))

        # Prepare discriminators for training method
        # Marginal discriminators
        D1_setup = DiscriminatorSetup("D1", D1, Utils.create_optim(D1.parameters(), opt),
                                      train_a, G_outputs, crop_fake=lambda x : x[:, 0:dataset.A_nc, :, :])
        D2_setup = DiscriminatorSetup("D2", D2, Utils.create_optim(D2.parameters(), opt),
                                      train_b, G_outputs, crop_fake=lambda x : x[:, dataset.A_nc:, :, :])
        D_setups = [D1_setup, D2_setup]

        # Dependency discriminators
        shuffle_batch_func = lambda x: Utils.shuffle_batch_dims(x, [dataset.A_nc])
        if opt.use_real_dep_disc:
            DP_setup = DependencyDiscriminatorSetup("DP", DP, Utils.create_optim(DP.parameters(), opt),
                                                    train_joint, shuffle_batch_func)
        else:
            DP_setup = None

        DQ_setup = DependencyDiscriminatorSetup("DQ", DQ,Utils.create_optim(DQ.parameters(), opt),
                                                G_outputs, shuffle_batch_func)
        D_dep_setups = [DependencyDiscriminatorPair(DP_setup, DQ_setup)]
    else:
        D = ConvDiscriminator(opt.loadSize, opt.loadSize, nc, opt.disc_channels).to(device)
        print(sum(p.numel() for p in D.parameters()))
        D_setups = [DiscriminatorSetup("D", D, Utils.create_optim(D.parameters(), opt), train_joint, G_outputs)]
        D_dep_setups = []

    # RUN TRAINING
    training.AdversarialTraining.train(opt, G, G_inputs, G_opt, D_setups, D_dep_setups, device, opt.log_path)
    torch.save(G.state_dict(), os.path.join(opt.experiment_path, "G"))

def eval(opt):
    device = Utils.get_device(opt.cuda)
    set_paths(opt)

    # Get test dataset
    dataset = get_aligned_dataset(opt, "val")
    nc = dataset.A_nc + dataset.B_nc

    # SETUP GENERATOR MODEL
    G = ConvGenerator(opt, opt.generator_channels, opt.loadSize, nc).to(device)
    G_noise = torch.distributions.uniform.Uniform(torch.Tensor([-1] * opt.nz), torch.Tensor([1] * opt.nz))

    # Prepare data sources that are a combination of real data and generator network, or purely from the generator network
    G_input_data = DataLoader(GeneratorInputDataset(None, G_noise), num_workers=int(opt.workers),
                              batch_size=opt.batchSize, shuffle=True)
    G_inputs = InfiniteDataSampler(G_input_data)
    G_outputs = TransformDataSampler(InfiniteDataSampler(G_input_data), G, device)
    G.load_state_dict(torch.load(os.path.join(opt.experiment_path, opt.eval_model)))
    G.eval()

    # EVALUATE
    # GENERATE EXAMPLES
    generate_images(G, G_inputs, opt.gen_path, 1000, device, lambda x: Utils.create_image_pair(x, dataset.A_nc, dataset.B_nc))

    # COMPUTE LS DISTANCE
    # Partition into test train and test test
    test_train_samples = int(0.8 * float(len(dataset)))
    test_test_samples = len(dataset) - test_train_samples
    print("VALIDATION SAMPLES: " + str(test_train_samples))
    print("TEST SAMPLES: " + str(test_test_samples))
    real_test_train_loader = DataLoader(Subset(dataset, range(test_train_samples)), num_workers=int(opt.workers), batch_size=opt.batchSize, shuffle=True, drop_last=True)
    real_test_test_loader = DataLoader(Subset(dataset, range(test_train_samples, len(dataset))), num_workers=int(opt.workers), batch_size=opt.batchSize)

    # Initialise classifier
    classifier_factory = lambda : ConvDiscriminator(opt.loadSize, opt.loadSize, nc, filters=opt.ls_channels, spectral_norm=False).to(device)
    # Compute metric
    losses = LS.compute_ls_metric(classifier_factory, real_test_train_loader, real_test_test_loader, G_outputs, opt.ls_runs, device)

    # WRITE RESULTS INTO CSV FOR LATER ANALYSIS
    file_existed = os.path.exists(os.path.join(opt.experiment_path, "LS.csv"))
    with open(os.path.join(opt.experiment_path, "LS.csv"), "a") as csv_file:
        writer = csv.writer(csv_file)
        model = "factorGAN" if opt.factorGAN else "gan"
        if not file_existed:
            writer.writerow(["LS", "Model", "Samples", "Dataset", "Samples_Validation","Samples_Test"])
        for val in losses:
            writer.writerow([val, model, opt.num_joint_samples, opt.dataset, test_train_samples, test_test_samples])

def get_opt():
    # COLLECT ALL CMD ARGUMENTS
    parser = training.TrainingOptions.get_parser()

    parser.add_argument('--dataset', type=str, default="edges2shoes",
                        help="Dataset to train on - can be cityscapes or edges2shoes (but other img2img datasets can be integrated easily")
    parser.add_argument('--num_joint_samples', type=int, default=1000,
                        help="Number of joint observations available for training normal gan/dependency discriminators")
    parser.add_argument('--loadSize', type=int, default=64,
                        help="Dimensions (no. of pixels) the dataset images are resampled to")
    parser.add_argument('--generator_channels', type=int, default=64,
                        help="Number of intial feature channels used in G. 64 was used in the paper")
    parser.add_argument('--disc_channels', type=int, default=32,
                        help="Number of intial feature channels used in each discriminator")


    # LS distance eval settings
    parser.add_argument('--ls_runs', type=int, default=10,
                        help="Number of LS Discriminator training runs for evaluation")
    parser.add_argument('--ls_channels', type=int, default=16,
                        help="Number of initial feature channels used for LS discriminator. 16 in the paper")


    opt = parser.parse_args()
    print(opt)

    # Set generator to sigmoid output
    opt.generator_activation = "sigmoid"

    return opt

if __name__ == "__main__":
    opt = get_opt()

    if not opt.eval:
        train(opt)
    eval(opt)