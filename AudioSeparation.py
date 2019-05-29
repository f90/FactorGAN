import torch
import random
import os
from torch.utils.data import DataLoader

import training.TrainingOptions
import training.AdversarialTraining
import Utils
from datasets.GeneratorInputDataset import GeneratorInputDataset
from datasets.InfiniteDataSampler import InfiniteDataSampler
from datasets.TransformDataSampler import TransformDataSampler
from datasets.AudioSeparationDataset import MUSDBDataset
from eval.SourceSeparation import produce_musdb_source_estimates
from models.generators.Unet import Unet
from training.DiscriminatorTraining import DiscriminatorSetup, DependencyDiscriminatorSetup, DependencyDiscriminatorPair
from models.discriminators.ConvDiscriminator import ConvDiscriminator

def set_paths(opt):
    # Set up paths and create folders
    opt.experiment_path = os.path.join(opt.out_path, "AudioSeparation", opt.experiment_name)
    opt.gen_path = os.path.join(opt.experiment_path, "gen")
    opt.log_path = os.path.join(opt.experiment_path, "logs")
    opt.estimates_path = os.path.join(opt.experiment_path, "source_estimates")
    Utils.make_dirs([opt.experiment_path, opt.gen_path, opt.log_path, opt.estimates_path])

def train(opt):
    Utils.set_seeds(opt)
    device = Utils.get_device(opt.cuda)
    set_paths(opt)

    if opt.num_joint_songs > 100:
        print("ERROR: Cannot train with " + str(opt.num_joint_songs) + " samples, dataset has only size of 100")
        return

    # Partition into paired and unpaired songs
    idx = [i for i in range(100)]
    random.shuffle(idx)

    # Joint samples
    dataset_train = MUSDBDataset(opt, idx[:opt.num_joint_songs], "paired")
    train_joint = InfiniteDataSampler(DataLoader(dataset_train, num_workers=int(opt.workers), batch_size=opt.batchSize, shuffle=True, drop_last=True))

    if opt.factorGAN == 1:
        # For marginals, take full dataset
        mix_dataset = MUSDBDataset(opt, idx, "mix")

        acc_dataset = MUSDBDataset(opt, idx, "accompaniment")
        acc_loader = InfiniteDataSampler(DataLoader(acc_dataset, num_workers=int(opt.workers), batch_size=opt.batchSize, shuffle=True, drop_last=True))

        vocal_dataset = MUSDBDataset(opt, idx, "vocals")
        vocal_loader = InfiniteDataSampler(DataLoader(vocal_dataset, num_workers=int(opt.workers), batch_size=opt.batchSize, shuffle=True, drop_last=True))
    else: # For normal GAN, take only few joint songs
        mix_dataset = MUSDBDataset(opt, idx[:opt.num_joint_songs], "mix")

    # SETUP GENERATOR MODEL
    G = Unet(opt, opt.generator_channels, 1, 1).to(device) # 1 input channel (mixture), 1 output channel (mask)
    G_noise = torch.distributions.uniform.Uniform(torch.Tensor([-1] * opt.nz), torch.Tensor([1] * opt.nz))
    G_opt = Utils.create_optim(G.parameters(), opt)

    # Prepare data sources that are a combination of real data and generator network, or purely from the generator network
    G_input_data = DataLoader(GeneratorInputDataset(mix_dataset, G_noise), num_workers=int(opt.workers),
                              batch_size=opt.batchSize, shuffle=True, drop_last=True)
    G_inputs = InfiniteDataSampler(G_input_data)
    G_filled_outputs = TransformDataSampler(InfiniteDataSampler(G_inputs), G, device)

    # SETUP DISCRIMINATOR(S)
    crop_mix = lambda x: x[:, 1:, :, :]  # Only keep sources, not mixture for dep discs
    if opt.factorGAN == 1:
        # Setup marginal disc networks
        D_voc = ConvDiscriminator(opt.input_height, opt.input_width, 1, opt.disc_channels).to(device)
        D_acc = ConvDiscriminator(opt.input_height, opt.input_width, 1, opt.disc_channels).to(device)

        D_acc_setup = DiscriminatorSetup("D_acc", D_acc, Utils.create_optim(D_acc.parameters(), opt), acc_loader,
                                      G_filled_outputs, crop_fake=lambda x : x[:,1:2,:,:])

        D_voc_setup = DiscriminatorSetup("D_voc", D_voc, Utils.create_optim(D_voc.parameters(), opt), vocal_loader,
                                      G_filled_outputs, crop_fake=lambda x : x[:,2:3,:,:])
        # Marginal discriminator
        D_setups = [D_acc_setup, D_voc_setup]

        # If our dep discriminators are only defined on classifier probabilites, integrate classification into discriminator network as first step
        if opt.use_real_dep_disc == 1:
            DP = ConvDiscriminator(opt.input_height, opt.input_width, 2, opt.disc_channels, spectral_norm=(opt.lipschitz_p == 1)).to(device)
        else:
            DP = lambda x : 0

        DQ = ConvDiscriminator(opt.input_height, opt.input_width, 2, opt.disc_channels).to(device)

        # Dependency discriminators
        shuffle_batch_func = lambda x: Utils.shuffle_batch_dims(x, 1) # Randomly mixes different sources together (e.g. accompaniment from one song with vocals from another)

        if opt.use_real_dep_disc:
            DP_setup = DependencyDiscriminatorSetup("DP", DP, Utils.create_optim(DP.parameters(), opt),
                                                    train_joint, shuffle_batch_func, crop_func=crop_mix)
        else:
            DP_setup = None

        DQ_setup = DependencyDiscriminatorSetup("DQ", DQ, Utils.create_optim(DQ.parameters(), opt),
                                                G_filled_outputs, shuffle_batch_func, crop_func=crop_mix)
        D_dep_setups = [DependencyDiscriminatorPair(DP_setup, DQ_setup)]
    else:
        D = ConvDiscriminator(opt.input_height, opt.input_width, 2, opt.disc_channels).to(device)

        D_setup = DiscriminatorSetup("D", D, Utils.create_optim(D.parameters(), opt),
                           train_joint, G_filled_outputs, crop_real=crop_mix, crop_fake=crop_mix)
        D_setups = [D_setup]
        D_dep_setups = []

    # RUN TRAINING
    training.AdversarialTraining.train(opt, G, G_inputs, G_opt, D_setups, D_dep_setups, device, opt.log_path)
    torch.save(G.state_dict(), os.path.join(opt.experiment_path, "G"))

def eval(opt):
    Utils.set_seeds(opt)
    device = Utils.get_device(opt.cuda)
    set_paths(opt)

    # GENERATOR
    # SETUP GENERATOR MODEL
    G = Unet(opt, opt.generator_channels, 1, 1).to(device)  # 1 input channel (mixture), 1 output channel (mask)
    G_noise = torch.distributions.uniform.Uniform(torch.Tensor([-1] * opt.nz), torch.Tensor([1] * opt.nz))
    G.load_state_dict(torch.load(os.path.join(opt.experiment_path, "G")))
    G.eval()

    #EVALUATE BY PRODUCING SOURCE ESTIMATE AUDIO AND SDR METRICS
    produce_musdb_source_estimates(opt, G, G_noise, opt.estimates_path, subsets="test")


def get_opt():
    # COLLECT ALL CMD ARGUMENTS
    parser = training.TrainingOptions.get_parser()

    parser.add_argument('--musdb_path', type=str, help="Path to MUSDB dataset")
    parser.add_argument('--preprocessed_dataset_path', type=str, help="Path to where the preprocessed dataset should be saved")

    parser.add_argument('--num_joint_songs', type=int, default=100,
                        help="Number of songs from which joint observations are available for training normal gan/dependency discriminators")
    parser.add_argument('--hop_size', type=int, default=256,
                        help="Hop size of FFT")
    parser.add_argument('--fft_size', type=int, default=512,
                        help="Size of FFT")
    parser.add_argument('--sample_rate', type=int, default=22050,
                        help="Resample input audio to this sample rate")
    parser.add_argument('--generator_channels', type=int, default=32,
                        help="Number of intial feature channels used in G. 32 was used in the paper")
    parser.add_argument('--disc_channels', type=int, default=32,
                        help="Number of intial feature channels used in each discriminator")

    opt = parser.parse_args()
    print(opt)

    opt.input_height = opt.fft_size // 2 # No. of freq bins for model
    opt.input_width = opt.input_height  // 2 # No of time frames

    print("Activating generator mask and sigmoid non-linearity for mask")
    opt.generator_mask = 1 # Use a mask for Unet output
    opt.generator_activation = "sigmoid" # Use sigmoid output for mask

    return opt

if __name__ == "__main__":
    opt = get_opt()

    if not opt.eval:
        train(opt)
    eval(opt)