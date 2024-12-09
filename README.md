# Multi-View Latent Diffusion Model (MV-LDM)

[![arXiv](https://img.shields.io/badge/arXiv-2403.16292-b31b1b.svg)](https://arxiv.org/abs/2403.16292)

This is an extension of the codebase for 

**MET3R: Measuring Multi-View Consistency in Generated Images** \
*Mohammad Asim, Christopher Wewer, Thomas Wimmer, Bernt Schiele, and Jan Eric Lenssen*

Check out the [project website here](https://geometric-rl.mpi-inf.mpg.de/met3r/).


## Installation

To get started, create a conda environment:

```bash

conda create -n diffsplat python=3.10
conda activate diffsplat

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt

```


<details>
<summary>Troubleshooting</summary>
<br>

If you face unrealistic CUDA out of memory issues (probably because of different GPU architectures during kernel compilation and training), try deinstalling the rasterizer and installing it with specified architectures:
```bash
pip uninstall diff-gaussian-rasterization
TORCH_CUDA_ARCH_LIST="6.0 7.0 7.5 8.0 8.6+PTX" pip install git+https://github.com/Chrixtar/latent-gaussian-rasterization
```
</details>

## Acquiring Datasets
Please move all dataset directories into a newly created `datasets` folder in the project root directory or modify the root path as part of the dataset config files in `config/dataset`.

### RealEstate10k
For experiments on RealEstate10k, we use the same dataset version and preprocessing into chunks as pixelSplat. Please refer to their codebase [here](https://github.com/dcharatan/pixelsplat#acquiring-datasets) for information about how to obtain the data.

## Acquiring Pre-trained Checkpoints
We provide two sets of checkpoints as part of our releases [here](https://github.com/Chrixtar/latentsplat/releases):
1. Pre-trained autoencoders and discriminators from [LDM](https://github.com/CompVis/latent-diffusion) adapted for finetuning within latentSplat. They serve as a starting point for latentSplat training. Please download the [pretrained.zip] and extract it in the project root directory for training from scratch.

2. Trained checkpoint of **MV-LDM** for RealEstate10k are available o Hugging Face at [asimbluemoon/mvldm-1.0](https://huggingface.co/asimbluemoon/mvldm-1.0/tree/main).

## Running the Code

### Sampling RealEstate-10K Scenes

The main entry point is `src/scripts/generate_mvldm.py`. Call it via:

```bash
python -m src.scripts.generate_mvldm +experiment=baseline model.anchors=4 checkpointing.load="<path-to-checkpoint>" mode=test dataset/view_sampler=evaluation dataset.view_sampler.index_path=assets/evaluation_index/re10k_video.json test.mode=video_anchor ind="<scene-index>" out_dir=./outputs/mvldm dataset.root="<path-to-re10k-dataset>"

```

This configuration requires a GPU with at least 40 GB of VRAM. 


### Training MV-LDM
Our code supports multi-GPU training. The above batch size is the per-GPU batch size.



## BibTeX

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <pre><code>@inproceedings{wewer24latentsplat,
    title     = {latentSplat: Autoencoding Variational Gaussians for Fast Generalizable 3D Reconstruction},
    author    = {Wewer, Christopher and Raj, Kevin and Ilg, Eddy and Schiele, Bernt and Lenssen, Jan Eric},
    booktitle = {arXiv},
    year      = {2024},
}</code></pre>
  </div>
</section>


