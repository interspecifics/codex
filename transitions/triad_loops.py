#
# 2021.08.16
# ----------
# triad loops by interpolation using tables
# 
#
import sys
#sys.path.append('/media/emme/EXP/CODEX/nn/Epoching_StyleGan2_Setup/stylegan2/')
sys.path.append('/media/emme/EXP/CODEX/nn/stylegan2/')

import pretrained_networks
import dnnlib
from dnnlib import tflib
from training import misc

from pathlib import Path
from PIL import Image
import pickle
import numpy as np
from random import randint

from tqdm import tqdm


AGLE7410 = [[ 564672,  462311, 130732],
            [ 80899 ,  836830, 754126],
            [ 603222,   80899, 505763],
            [ 179034,  635874,  85495],
            [ 120263,  525240, 566243],
            [  99935,   53410, 936481],
            [ 527801,  527801, 166322],
            [ 762062,  590479, 762062],
            [ 431409,  130526, 639155],
            [1096609,  369232, 572825],
            [  86202,  954621, 299727],
            [ 332117,  742173, 170040],
            [ 493869,  281323, 752458],
            [ 170762,  151675, 423005]]

AGDR7410 = [[ 407613,  959696, 1056509],
            [1090009,  313666,   75658],
            [ 405324,  429550,  843031],
            [1047198,  382637,   86846],
            [ 697683,  201989,  182063],
            [  70055,  771623,  565798],
            [ 727051,  405842,  177703],
            [ 583654, 1035142,  163912],
            [1033996,  382013,   48556],
            [ 251980,   70055,  988111],
            [ 446863,   86759, 1055181],
            [ 258217,  967317,  299239],
            [ 619338,   22038,  481535],
            [ 332653, 1005402,  412764],
            [1055232,   82862,  404683],
            [ 931107,   62091,  674916],
            [ 244792,  574713,  401436],
            [ 183567,  274118,  863640],
            [1061495,  561191, 1022351],
            [ 290845,  409867,   50724],
            [ 381867,   99382,  294165],
            [ 587652,  305912,  545661],
            [ 755109,  284888,  695272],
            [ 379501,  485782,  970170],
            [ 273446,  721477,  872590],
            [1002713,  710540,  428579],
            [1079252,  769541,  146687],
            [ 752978,  997228,  874663],
            [ 483295,  104191,  692816],
            [ 832591,  613656,   19945],
            [ 889208,   33901,  706090],
            [ 791076,  447022,  761964],
            [ 537464,  414789,  972697],
            [ 577309,   72954,  733462],
            [ 466400,  216574,  860414],
            [ 393080,  140365,  173943]]


AGCH7410 = [[ 572051,  470462,  622981],
            [1096837, 1006203,  603097],
            [ 662639,  588487,  997748],
            [ 748111,  146028,  428094],
            [  10543,  868869,  878993],
            [ 112931,  239218,  895789],
            [ 325189,  914284,  422842],
            [ 676133,  612042,  349759],
            [ 961115,  835912,  617002],
            [ 676328,  244189,  325189],
            [ 605468,  792570,  374467],
            [ 273020,  137931,  432868],
            [  78544,  955919,  897764],
            [ 190846,  297885,  691920],
            [ 786942,  115748,  369380],
            [ 654902,  827575,  153545],
            [ 937570,  479961, 1091390],
            [ 706988, 1012486,  473370],
            [ 604432,  213589,  882343],
            [ 551196,  778641,    9775],
            [ 110938,  127574,  474223],
            [1018400,  419509,  361234],
            [ 443820,  176713,  693991],
            [ 911444, 1072879,  107308],
            [ 703436,  140544,    4275],
            [ 460294, 1035945,  965686],
            [1030409,   86693,  981122],
            [ 147814,  959874,  852187],
            [ 784964,  490595, 1106253],
            [ 637977,  959913,  958499],
            [ 743532,  997734,  545863]]


####################   #   #   #   #   #   #  #  ################################
db_name = "AGDR"        # cambiar segun familia
numb = 7410             # cambiar segun ¿generación?
actual_fam = AGDR7410   # cambiar segun variable de lista (ver arriba)
                        # cambiar directorio base del modelo, "/media/emme/EXP/CODEX/nn/stylegan2/pretrained/stylegan2_HY_XF/agua7404+dr/"
model_path = '/media/emme/EXP/CODEX/nn/stylegan2/pretrained/stylegan2_HY_XF/agua7404+dr/network-snapshot-00{}.pkl'.format(numb)
                        # cambiar directorio base de salida "/media/emme/EXP/CODEX/AGUA_HY/"
outpathname = '/media/emme/EXP/CODEX/AGUA_HY/{}_00{}_loops'.format(db_name, numb)
##################################################################################

# set paths for folders
output_gifs_path = Path(outpathname)
triad_path = Path(outpathname)
# Make Output Gifs folder if it doesn't exist.
if not output_gifs_path.exists():
    output_gifs_path.mkdir()

fps = 12
results_size = 512

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

# Same but saving the file
def save_image_random(rand_seed):
    rnd = np.random.RandomState(rand_seed)
    z = rnd.randn(1, *Gs.input_shape[1:])
    tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars})
    images = Gs.run(z, None, **Gs_kwargs)
    ffnn = '/i_{}_00{}'.format(db_name, numb)
    Image.fromarray(images[0], 'RGB').save(dnnlib.make_run_dir_path(str(triad_path)+ ffnn +'_s%08d.png' % rand_seed))

def save_image_from_z(z, prefix):
    images = Gs.run(z, None, **Gs_kwargs)
    Image.fromarray(images[0], 'RGB').save(dnnlib.make_run_dir_path(str(triad_path)+'/'+prefix+'.png'))

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
    save_name = triad_path/'{}.gif'.format(prefix)
    all_imgs[0].save(save_name, save_all=True, append_images=all_imgs[1:], duration=1000/fps, loop=0)

# LOAD
Gs, noise_vars, Gs_kwargs = load_model()


# --------------------------------------
def gen_from_seed_variation(fixed_seed = None):
    if fixed_seed == None:
        ra = randint(1, 1111111)
    else:
        ra = fixed_seed
    p1 = 2.5
    p2 = 1.0
    prefix = '{}_{}_{}_{}'.format(db_name.lower(),ra,p1,p2)
    images, latent_code_A = generate_image_random(ra)
    save_image_random(ra)
    latent_code_B = latent_code_A.copy()
    latent_code_C = latent_code_A.copy() #-1,+3, -3,+2
    latent_code_B = latent_code_B + p1
    latent_code_C = latent_code_C + p2
    imagesB = generate_image_from_z(latent_code_B)
    imagesC = generate_image_from_z(latent_code_C)
    imgB = Image.fromarray(imagesB[0])
    imgC = Image.fromarray(imagesC[0])
    #interpolated_latent_code = linear_interpolate(latent_code_B, latent_code_C, 0.33)
    #images = generate_image_from_z(interpolated_latent_code)
    make_latent_interp_animation(latent_code_B, latent_code_C, imgB, imgC, 720, prefix)


def gen_from_seed_to_random(fixed_seed = 111111):
    if fixed_seed == None:
        ra = randint(1, 1111111)
    else:
        ra = fixed_seed
    imagesA, latent_code_A = generate_image_random(ra)
    save_image_random(ra)
    rb = randint(1, 1111111)
    imagesB, latent_code_B = generate_image_random(rb)
    save_image_random(rb)
    prefix = '{}_{}_{}'.format(db_name.lower(),ra,rb)
    latent_code_C = latent_code_B * latent_code_A - 2
    #print("LATENT_CODE_1: {}".format(latent_code1))
    #latent_code_C = latent_code_B + latent_code_A
    #latent_code_C = latent_code_A - latent_code_B**2
    imagesC = generate_image_from_z(latent_code_C)
    save_image_from_z(latent_code_C, prefix)
    imgA = Image.fromarray(imagesA[0])
    imgC = Image.fromarray(imagesC[0])
    #interpolated_latent_code = linear_interpolate(latent_code_B, latent_code_C, 0.33)
    #images = generate_image_from_z(interpolated_latent_code)
    make_latent_interp_animation(latent_code_C, latent_code_A, imgC, imgA, 240, prefix)


def gen_from_seed_to_seed(sa = 111111, sb=111112):
    if (sa == None or sb==None):
        ra = randint(1, 1111111)
        rb = randint(1, 1111111)
    else:
        ra = sa
        rb = sb
    imagesA, latent_code_A = generate_image_random(ra)
    save_image_random(ra)
    imagesB, latent_code_B = generate_image_random(rb)
    save_image_random(rb)
    prefix = '{}_{}_{}'.format(db_name.lower(),ra,rb)
    imgA = Image.fromarray(imagesA[0])
    imgB = Image.fromarray(imagesB[0])
    #interpolated_latent_code = linear_interpolate(latent_code_B, latent_code_C, 0.33)
    #images = generate_image_from_z(interpolated_latent_code)
    make_latent_interp_animation(latent_code_B, latent_code_A, imgB, imgA, 720, prefix)


def get_random_sample_grid():
    img_grid_size = (10, 8)
    drange_net    = [-1,1]
    grid_latents = np.random.randn(np.prod(img_grid_size), 512)
    image_grid = Gs.run(grid_latents, None, **Gs_kwargs)
    misc.save_image_grid(image_grid, dnnlib.make_run_dir_path('samplegrid.png'), drange=drange_net, grid_size=img_grid_size)


def get_sample_list(list_size=200):
    latents_list = np.random.randn(list_size, 512)
    print (latents_list.shape)
    latents_list_2 = latents_list.tolist()
    fn = str(output_gifs_path)+'/db.txt'
    f = open(fn, 'w+')
    for i in range(len(latents_list)):
        prefix = 'r_{}_{0:04d}'.format(db_name.lower(),i)
        print('going print {}'.format(latents_list[i]))
        save_image_from_z(np.array([2*latents_list[i]]), prefix)
        vals = ['{:2.4f}'.format(v) for v in latents_list_2[i]]
        f.write(','.join(vals)+'\n')
    print("printed file: {}".format(fn))
    f.close()


def get_better_sample_list(list_size=200):
    fn = str(output_gifs_path)+'/db_{}_00{}.txt'.format(db_name,numb)
    f = open(fn, 'w+')
    for i in range(list_size):
        ra = randint(1, 1111111)
        print('vector from seed: {}'.format(ra))
        imagesA, latent_code = generate_image_random(ra)
        save_image_random(ra)
        vals = ['{:2.8f}'.format(v) for v in latent_code[0].tolist()]
        f.write(','.join(vals)+'\n')
    print("printed file: {}".format(fn))
    f.close()


def gen_variations_from_seed(fixed_seed = None):
    if fixed_seed == None:
        ra = randint(1, 1111111)
    else:
        ra = fixed_seed
    ixes = [[-1.0, 0.0],[0.0,-1.0],[1.0, 0.0],[2.5, 1.0]]
    for ix in ixes:
        p1, p2 = ix
        prefix = '{}_{}_{}_{}'.format(db_name.lower(),ra,p1,p2)
        images, latent_code_A = generate_image_random(ra)
        save_image_random(ra)
        latent_code_B = latent_code_A.copy()
        latent_code_C = latent_code_A.copy() #-1,+3, -3,+2
        latent_code_B = latent_code_B + p1
        latent_code_C = latent_code_C + p2
        imagesB = generate_image_from_z(latent_code_B)
        imagesC = generate_image_from_z(latent_code_C)
        imgB = Image.fromarray(imagesB[0])
        imgC = Image.fromarray(imagesC[0])
        #interpolated_latent_code = linear_interpolate(latent_code_B, latent_code_C, 0.33)
        #images = generate_image_from_z(interpolated_latent_code)
        make_latent_interp_animation(latent_code_B, latent_code_C, imgB, imgC, 720, prefix)
    print("ending generation")
# --------------------------------------


def triseq(a,b,c):
    gen_from_seed_to_seed(a, b)
    gen_from_seed_to_seed(b, c)
    gen_from_seed_to_seed(c, a)



# -- generate a list of loop-triads in out_dir
for aa,bb,cc in actual_fam:   
    triad_pathname = outpathname + "/{}_{}_{}".format(aa, bb, cc)
    triad_path = Path(triad_pathname)
    if not triad_path.exists():
        triad_path.mkdir()
    triseq(aa, bb, cc)
    print ("[._.] {} - {} - {}".format(aa,bb,cc))
print ("|-- \t\t.[._.].b")



