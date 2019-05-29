import os
import torchvision.utils as vutils

def generate_images(generator, generator_inputs, out_path, num_images, device, transform=None):
    # Create subfolder if it doesn't exist
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Generate images and save
    idx = 0
    while idx < num_images:
        gen_input = next(generator_inputs)
        gen_input = [item.to(device) for item in gen_input]
        sample_batch = generator(gen_input)
        if transform is not None:
            sample_batch = transform(sample_batch)
        for sample in sample_batch:
            save_as_image(sample, os.path.join(out_path, "gen_" + str(idx) + ".png"))
            idx += 1

def save_as_image(tensor, path):
    vutils.save_image(tensor, path, normalize=True)