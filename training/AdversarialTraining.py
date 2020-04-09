import os

from tqdm import tqdm

from training.DiscriminatorTraining import *
from torch.utils.tensorboard import SummaryWriter

def train(cfg, G, G_input, G_opt, D_marginal_setups, D_dep_pairs, device, logdir):
    print("START TRAINING! Writing logs to " + logdir)
    writer = SummaryWriter(logdir)

    # Create expression for overall discriminator output given a complete fake sample
    # Marginal disc sum output
    marginal_sum = lambda y: sum(setup.D(setup.crop_fake(y)) for setup in D_marginal_setups)
    if cfg.factorGAN == 1:
        # Dep disc sum output
        dep_sum = lambda y: sum(disc_pair.get_comb_disc()(y) for disc_pair in D_dep_pairs)
        jointD = lambda y: marginal_sum(y) + dep_sum(y)
    else:
        jointD = marginal_sum

    # START NORMAL TRAINING
    for epoch in range(cfg.epochs):
        for i in tqdm(range(cfg.epoch_iter)):
            total_it = epoch * cfg.epoch_iter + i
            # If dependency GAN active, train marginal discriminators here from both extra data and main data
            for j in range(cfg.disc_iter):  # No. of disc iterations
                # Train marginal discriminators
                for D_setup in D_marginal_setups:
                    errD, correct, _, _ = get_marginal_disc_output(D_setup, device, backward=True, zero_gradients=True)
                    if j==cfg.disc_iter-1: writer.add_scalar(D_setup.name + "_acc", correct, total_it)
                    D_setup.optim.step()

                if cfg.factorGAN == 1:
                    # Additionally train dependency discriminators
                    for D_dep_pair in D_dep_pairs:
                        # Train REAL dependency discriminator
                        if cfg.use_real_dep_disc == 1:
                            # Training step for real dep disc
                            errD, correct, _, _ = get_dep_disc_output(D_dep_pair.real_disc, device, backward=True,zero_gradients=True)
                            D_dep_pair.real_disc.optim.step()

                            # Logging for last discriminator update
                            if j == cfg.disc_iter - 1: writer.add_scalar(D_dep_pair.real_disc.name + "_acc", correct, total_it)
                            if j == cfg.disc_iter - 1: writer.add_scalar(D_dep_pair.real_disc.name + "_errD", errD, total_it)

                        # Train FAKE dependency discriminator. Use combined output of real and fake dependency discs for regularisation purposes => Fake dep. disc needs to ensure its gradients stay close to the real dep. ones
                        errD, correct, _, _ = get_dep_disc_output(D_dep_pair.fake_disc, device, backward=True, zero_gradients=True)
                        D_dep_pair.fake_disc.optim.step()

                        # Logging for last discriminator update
                        if j == cfg.disc_iter - 1: writer.add_scalar(D_dep_pair.fake_disc.name + "_acc", correct, total_it)
                        if j == cfg.disc_iter - 1: writer.add_scalar(D_dep_pair.fake_disc.name + "_errD", errD, total_it)

            ############################
            # (2) Update G network:
            ###########################

            G.zero_grad()

            # Get fake samples from generator
            gen_input = G_input.__next__()
            gen_input = [item.to(device) for item in gen_input]
            fake_sample = G(gen_input)

            #TODO Produce log outputs for all tasks here?

            # Get setup information from first marginal discriminator (which is the only one in normal GAN training)
            real_label = D_marginal_setups[0].real_label
            criterion = D_marginal_setups[0].criterion

            label = torch.full((cfg.batchSize,), real_label, device=device)  # fake labels are real for generator cost
            disc_output = jointD(fake_sample)
            writer.add_scalar("probG", torch.mean(torch.nn.Sigmoid()(disc_output)), total_it)
            if cfg.objective == "JSD":
                errG = criterion()(disc_output, label)  # Normal JSD
            elif cfg.objective == "KL":
                errG = -torch.mean(disc_output)  # KL[q|p]
            else:
                raise NotImplementedError

            writer.add_scalar("errG", errG, i)

            errG.backward()
            G_opt.step()

        print("EPOCH FINISHED")

        model_output_path = os.path.join(cfg.experiment_path, "G_" + str(epoch))
        print("Saving generator at " + model_output_path)
        torch.save(G.state_dict(), model_output_path)

    # FINISHED
    print("TRAINING FINISHED")