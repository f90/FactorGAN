import torch
from tensorboardX import SummaryWriter

import os
from torch.utils.data import DataLoader, Subset

import training.TrainingOptions
import training.AdversarialTraining
import Utils
from datasets.GeneratorInputDataset import GeneratorInputDataset
from datasets.InfiniteDataSampler import InfiniteDataSampler
from datasets.TransformDataSampler import TransformDataSampler
from datasets.image2image import get_aligned_dataset
from eval.Cityscapes import get_L2, get_pixel_acc
from eval.Visualisation import generate_images
from models.generators.Unet import Unet
from training.DiscriminatorTraining import DiscriminatorSetup, DependencyDiscriminatorSetup, DependencyDiscriminatorPair
from models.discriminators.ConvDiscriminator import ConvDiscriminator
from datasets.CropDataset import CropDataset

def set_paths(opt):
    # Set up paths and create folders
    opt.experiment_path = os.path.join(opt.out_path, "Image2Image_" + str(opt.dataset), opt.experiment_name)
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

    if opt.num_joint_samples > len(dataset):
        print("WARNING: Cannot train with " + str(opt.num_joint_samples) + " samples, dataset has only size of " + str(len(dataset))+ ". Using full dataset!")
        opt.num_joint_samples = len(dataset)

    # Joint samples
    dataset_train = Subset(dataset, range(opt.num_joint_samples))
    train_joint = InfiniteDataSampler(
        DataLoader(dataset_train, num_workers=int(opt.workers), batch_size=opt.batchSize, shuffle=True, drop_last=True))

    if opt.factorGAN == 1:
        # For marginals, take full dataset and crop
        a_dataset = CropDataset(dataset, lambda x : x[0:dataset.A_nc, :, :])
        b_dataset = CropDataset(dataset, lambda x : x[dataset.A_nc:, :, :])
        train_b = InfiniteDataSampler(DataLoader(b_dataset, num_workers=int(opt.workers), batch_size=opt.batchSize, shuffle=True, drop_last=True))

        generator_input_data = a_dataset
    else:
        generator_input_data = CropDataset(dataset_train, lambda x : x[0:dataset.A_nc, :, :])

    # SETUP GENERATOR MODEL
    G = Unet(opt, opt.generator_channels, dataset.A_nc, dataset.B_nc).to(device)
    G_noise = torch.distributions.uniform.Uniform(torch.Tensor([-1] * opt.nz), torch.Tensor([1] * opt.nz))
    G_opt = Utils.create_optim(G.parameters(), opt)
    # Prepare data sources that are a combination of real data and generator network, or purely from the generator network
    G_input_data = DataLoader(GeneratorInputDataset(generator_input_data, G_noise), num_workers=int(opt.workers),
                              batch_size=opt.batchSize, shuffle=True, drop_last=True)
    G_inputs = InfiniteDataSampler(G_input_data)
    G_filled_outputs = TransformDataSampler(InfiniteDataSampler(G_input_data), G, device)

    # SETUP DISCRIMINATOR(S)
    if opt.factorGAN == 1:
        # Setup disc networks
        D2 = ConvDiscriminator(opt.loadSize, opt.loadSize, dataset.B_nc).to(device)
        # If our dep discriminators are only defined on classifier probabilites, integrate classification into discriminator network as first step
        if opt.use_real_dep_disc == 1:
            DP = ConvDiscriminator(opt.loadSize, opt.loadSize, nc, spectral_norm=(opt.lipschitz_p == 1)).to(device)
        else:
            DP = lambda x : 0

        DQ = ConvDiscriminator(opt.loadSize, opt.loadSize, nc).to(device)

        # Prepare discriminators for training method
        # Marginal discriminator
        D_setups = [DiscriminatorSetup("D2", D2, Utils.create_optim(D2.parameters(), opt),
                                      train_b, G_filled_outputs, crop_fake=lambda x: x[:, dataset.A_nc:, :, :])]
        # Dependency discriminators
        shuffle_batch_func = lambda x: Utils.shuffle_batch_dims(x, [dataset.A_nc])
        if opt.use_real_dep_disc:
            DP_setup = DependencyDiscriminatorSetup("DP", DP, Utils.create_optim(DP.parameters(), opt),
                                                    train_joint, shuffle_batch_func)
        else:
            DP_setup = None

        DQ_setup = DependencyDiscriminatorSetup("DQ", DQ, Utils.create_optim(DQ.parameters(), opt),
                                                G_filled_outputs, shuffle_batch_func)
        D_dep_setups = [DependencyDiscriminatorPair(DP_setup, DQ_setup)]
    else:
        D = ConvDiscriminator(opt.loadSize, opt.loadSize, nc).to(device)
        print(sum(p.numel() for p in D.parameters()))

        D_setup = DiscriminatorSetup("D", D, Utils.create_optim(D.parameters(), opt),
                       train_joint, G_filled_outputs)
        D_setups = [D_setup]
        D_dep_setups = []

    # RUN TRAINING
    training.AdversarialTraining.train(opt, G, G_inputs, G_opt, D_setups, D_dep_setups, device, opt.log_path)
    torch.save(G.state_dict(), os.path.join(opt.experiment_path, "G"))

def eval(opt):
    Utils.set_seeds(opt)
    device = Utils.get_device(opt.cuda)
    set_paths(opt)

    # DATASET
    dataset = get_aligned_dataset(opt, "val")
    input_dataset = CropDataset(dataset, lambda x: x[0:dataset.A_nc, :, :])

    # GENERATOR
    G = Unet(opt, opt.generator_channels, dataset.A_nc, dataset.B_nc).to(device)
    G_noise = torch.distributions.uniform.Uniform(torch.Tensor([-1] * opt.nz), torch.Tensor([1] * opt.nz))
    G.load_state_dict(torch.load(os.path.join(opt.experiment_path, "G")))
    G.eval()

    # EVALUATE: Generate some images using test set and noise as conditional input
    G_input_data = DataLoader(GeneratorInputDataset(input_dataset, G_noise), num_workers=int(opt.workers),
                              batch_size=opt.batchSize, shuffle=False)
    G_inputs = InfiniteDataSampler(G_input_data)

    generate_images(G, G_inputs, opt.gen_path, 100, device, lambda x : Utils.create_image_pair(x, dataset.A_nc, dataset.B_nc))

    # EVALUATE for Cityscapes
    if opt.dataset == "cityscapes":
        writer = SummaryWriter(opt.log_path)
        val_input_data = DataLoader(dataset, num_workers=int(opt.workers),batch_size=opt.batchSize)

        pixel_error = get_pixel_acc(opt, device, G, val_input_data, G_noise)
        print("VALIDATION PERFORMANCE Pixel: " + str(pixel_error))
        writer.add_scalar("val_pix", pixel_error)

        L2_error = get_L2(opt, device, G, val_input_data, G_noise)
        print("VALIDATION PERFORMANCE L2: " + str(L2_error))
        writer.add_scalar("val_L2", L2_error)


def get_opt():
    # COLLECT ALL CMD ARGUMENTS
    parser = training.TrainingOptions.get_parser()

    parser.add_argument('--dataset', type=str, default="cityscapes",
                        help="Dataset to train on - can be cityscapes or edges2shoes (but other img2img datasets can be integrated easily")
    parser.add_argument('--num_joint_samples', type=int, default=1000,
                        help="Number of joint observations available for training normal gan/dependency discriminators")
    parser.add_argument('--loadSize', type=int, default=128,
                        help="Dimensions (no. of pixels) the dataset images are resampled to")
    parser.add_argument('--generator_channels', type=int, default=32,
                        help="Number of intial feature channels used in G. 64 was used in the paper")

    opt = parser.parse_args()
    print(opt)

    # Set generator to sigmoid output
    opt.generator_activation = "sigmoid"

    # Square images => loadSize determines width and height
    opt.input_width = opt.loadSize
    opt.input_height = opt.loadSize

    return opt

if __name__ == "__main__":
    opt = get_opt()

    if not opt.eval:
        train(opt)
    eval(opt)