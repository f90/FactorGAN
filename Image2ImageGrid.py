# This script will train image segmentation models in different dataset configurations as in the paper
import Image2Image

opt = Image2Image.get_opt()

for dataset_name in ["cityscapes", "edges2shoes"]: # Iterate over datasets, more datasets could be added here like maps
    opt.dataset = dataset_name
    for num_joint_samples in [100, 1000, 10000]: # Try for different amount of paired samples
        # Apply settings
        print(str(num_joint_samples) + " joint samples")
        opt.num_joint_samples = num_joint_samples

        print("Training GAN")
        opt.experiment_name = str(num_joint_samples) + "_joint_GAN"
        opt.factorGAN = 0
        Image2Image.train(opt)

        print("Training factorGAN")
        opt.experiment_name = str(num_joint_samples) + "_joint_factorGAN"
        opt.factorGAN = 1
        Image2Image.train(opt)