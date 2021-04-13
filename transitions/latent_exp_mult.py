# 
# 
# 2021.04.06
# ----------
# codex virtualis
# multiple images transition sequences from pretrained model
#
#
import sys
sys.path.append('./stylegan2/')

import pretrained_networks
import dnnlib
from dnnlib import tflib

from pathlib import Path
from PIL import Image
import pickle
import numpy as np

#import ipywidgets as widgets
from tqdm import tqdm

model_path = 'network-snapshot-009925.pkl'
fps = 20
results_size = 1024

output_gifs_path = Path('output_gifs')
# Make Output Gifs folder if it doesn't exist.
if not output_gifs_path.exists():
    output_gifs_path.mkdir()

# Code to load the StyleGAN2 Model
def load_model():
    _G, _D, Gs = pretrained_networks.load_networks(model_path)
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    return Gs, noise_vars, Gs_kwargs

# Generate images given a random seed (Integer)
def generate_image_random(rand_seed):
    rnd = np.random.RandomState(rand_seed)
    z = rnd.randn(1, *Gs.input_shape[1:])
    tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars})
    images = Gs.run(z, None, **Gs_kwargs)
    return images, z

# Generate images given a latent code ( vector of size [1, 512] )
def generate_image_from_z(z):
    images = Gs.run(z, None, **Gs_kwargs)
    return images

#latent_code1[0][:5]
#latent_code2[0][:5]

def linear_interpolate(code1, code2, alpha):
    return code1 * alpha + code2 * (1 - alpha)

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def make_latent_interp_animation(code1, code2, img1, img2, num_interps, prefix):
    step_size = 1.0/num_interps
    all_imgs = []
    amounts = np.arange(0, 1, step_size)
    for alpha in tqdm(amounts):
        interpolated_latent_code = linear_interpolate(code1, code2, alpha)
        images = generate_image_from_z(interpolated_latent_code)
        #interp_latent_image = Image.fromarray(images[0]).resize((400, 400))
        interp_latent_image = Image.fromarray(images[0])
        frame = interp_latent_image
        #frame = get_concat_h(img1, interp_latent_image)
        #frame = get_concat_h(frame, img2)
        all_imgs.append(frame)
    save_name = output_gifs_path/'{}.gif'.format(prefix)
    all_imgs[0].save(save_name, save_all=True, append_images=all_imgs[1:], duration=1000/fps, loop=0)

# LOAD
Gs, noise_vars, Gs_kwargs = load_model()




# Ask the generator to make an output, given a random seed number: 42
for i in range(11,21):
    r1 = i
    r2 = 10 #8023,8045,8004,8016
    nnn = 'ls9925_{}-{}'.format(r1, r2)
    images, latent_code1 = generate_image_random(r1)
    image1 = Image.fromarray(images[0])
    #image1 = Image.fromarray(images[0]).resize((results_size, results_size))
    images, latent_code2 = generate_image_random(r2)
    image2 = Image.fromarray(images[0])
    #image2 = Image.fromarray(images[0]).resize((results_size, results_size))
    #latent_code1.shape
    #latent_code2.shape
    interpolated_latent_code = linear_interpolate(latent_code1, latent_code2, 0.3)
    #interpolated_latent_code.shape
    # interpolation
    images = generate_image_from_z(interpolated_latent_code)
    #Image.fromarray(images[0]).resize((results_size, results_size))
    # make
    make_latent_interp_animation(latent_code1, latent_code2, image1, image2, 400, nnn)


"""
#python run_projector.py project-real-images --data-dir=/media/emme/EXP/CODEX/nn/stylegan2/datasets --dataset=micro --network=/media/emme/EXP/CODEX/nn/stylegan2/results/00022-stylegan2-micro-1gpu-config-e/network-snapshot-007000.pkl
#python run_generator.py generate-images --network=/media/emme/EXP/CODEX/nn/stylegan2/results/00022-stylegan2-micro-1gpu-config-e/network-snapshot-007000.pkl \
#    --seeds=6000-6025 --truncation-psi=0.5
"""