import argparse
import numpy as np

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default=str(np.random.randint(0, 100000)), help='experiment name')
    parser.add_argument('--out_path', type=str, default="out", help="Output path")

    parser.add_argument('--eval', action='store_true', help='Perform evaluation instead of training')
    parser.add_argument('--eval_model', type=str, default='G', help='Name of generator checkpoint file to load for evaluation')

    parser.add_argument('--epochs', type=int, default=40, help='number of epochs to train for')
    parser.add_argument('--epoch_iter', type=int, default=5000, help="Number of generator updates per epoch")

    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate, default=0.0002')
    parser.add_argument('--L2', type=float, default=0.0, help='L2 regularisation for discriminators (except joint p dependency discriminator')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')

    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--seed', type=int, default=1337, help='manual seed')

    # Generator settings
    parser.add_argument('--nz', type=int, default=50, help='size of the latent z vector')
    parser.add_argument('--factorGAN', type=int, default=0, help="Activate FactorGAN instead of normal GAN")

    parser.add_argument('--disc_iter', type=int, default=2, help="Number of discriminator(s) iteration per generator update")
    parser.add_argument('--objective', type=str, default="JSD", help="JSD or KL as generator objective")

    # Dependency settings
    parser.add_argument('--lipschitz_q', type=int, default=1, help="Spectral norm regularisation for fake dependency discriminators")
    parser.add_argument('--lipschitz_p', type=int, default=1, help="Spectral norm regularisation for real dependency discriminators")
    parser.add_argument('--use_real_dep_disc', type=int, default=1, help="1 to use the dependency discriminator on real data normally, 0 to not use it and set the output to zero, assuming our real data dimensions are independent")

    # Data loading settings
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
    parser.add_argument('--batchSize', type=int, default=25, help='input batch size')

    return parser