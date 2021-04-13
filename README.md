# codex
Codex Virtualis

# 1. exploring the latent space

1. install cuda 10.0

    https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal


2. install cudNN 7.5 for cuda 10.0

    https://developer.nvidia.com/rdp/cudnn-archive


3. with anaconda create new env 

    $ conda create -n style python=3.6

    $ conda install -c conda-forge tensorflow-gpu=1.14

    $ conda activate style 

    $ pip install tensorflow-gpu==1.14

    $ pip install requests Pillow, tqdm



4. install visual studio 2017 from

    https://my.visualstudio.com/Downloads?q=visual%20studio%202017&wt.mc_id=o~msft~vscom~older-downloads



5. clonar este repo

    $ git clone https://github.com/interspecifics/codex

    cd codex/transitions/

    descargar modelo de https://drive.google.com/drive/folders/1TnW5E4O9KeZ5MvRjJVgD7biDU0Aq4-j5?usp=sharing

    colocar archivo .pkl en codex/transitions/



5. update route for VStudio in codex/transitions/stylegan2/dnnlib/tflib/custom_ops.py:


    compiler_bindir_search_path = [

        'C:/Program Files (x86)/Microsoft Visual Studio/2017/Community/VC/Tools/MSVC/14.16.27023/bin/Hostx64/x64',

        #'C:/Program Files (x86)/Microsoft Visual Studio/2017/Community/VC/Tools/MSVC/14.14.26428/bin/Hostx64/x64',

        'C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Tools/MSVC/14.23.28105/bin/Hostx64/x64',

        'C:/Program Files (x86)/Microsoft Visual Studio 14.0/vc/bin',

        ]


8. edit random seeds for vector generation latent anim generator codex/transitions/latent_exp.py


    r1 = 12

    r2 = 10 #8023,8045,8004,8016


9. run 

    $ python latent_exp.py
