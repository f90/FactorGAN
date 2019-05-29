# Run ImagePairs experiment for multiple datasets and number of paired samples, using GAN and FactorGAN

import ImagePairs
opt = ImagePairs.getImageOpt()

for dataset_name in ["cityscapes", "edges2shoes"]:
    opt.dataset = dataset_name
    for num_joint_samples in [100, 1000, 10000]:

        # Apply settings
        print(str(num_joint_samples) + " joint samples")
        opt.num_joint_samples = num_joint_samples

        print("Training GAN")
        opt.experiment_name = str(num_joint_samples) + "_joint_GAN"
        opt.factorGAN = 0
        ImagePairs.train(opt)

        print("Training factorGAN")
        opt.experiment_name = str(num_joint_samples) + "_joint_factorGAN"
        opt.factorGAN = 1
        ImagePairs.train(opt)