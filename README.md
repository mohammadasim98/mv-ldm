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

Trained checkpoint of **MV-LDM** for RealEstate10k are available on Hugging Face at [asimbluemoon/mvldm-1.0](https://huggingface.co/asimbluemoon/mvldm-1.0/tree/main).

## Running the Code

### Sampling RealEstate-10K Scenes

The main entry point is `src/scripts/generate_mvldm.py`. Call it via:

> [!IMPORTANT] 
> Sampling requires a GPU with at least 16 GB of VRAM.

```bash
python -m src.scripts.generate_mvldm +experiment=baseline \
  mode=test \
  dataset.root="<root-path-to-re10k-dataset>" \
  scene_id="<scene-id>" \ 
  checkpointing.load="<path-to-checkpoint>" \
  dataset/view_sampler=evaluation \
  dataset.view_sampler.index_path=assets/evaluation_index/re10k_video.json \
  test.sampling_mode=anchored \
  test.num_anchors_views=4 \
  test.output_dir=./outputs/mvldm 
```

> [!NOTE]
> ```scene_id="<scene-id>"``` either defines the specific integer index which refers to an ID of a scene ordered as in ```assets/evaluation_index/re10k_video.json``` or the sequence ID as a string e.g. ```"2d3f982ada31489c"```.
> ```python
> scene_id=25
> scene_id="2d3f982ada31489c"
> ```
### Training MV-LDM
Our code supports multi-GPU training. The above batch size is the per-GPU batch size.

> [!IMPORTANT] 
> Training requires a GPU with at least 40 GB of VRAM.


```bash
python -m src.main +experiment=baseline \
  mode=train \
  dataset.root="<root-path-to-re10k-dataset>" \
  hydra.run.dir="<runtime-dir>" \
  hydra.job.name=train
```

In case of memory issues during training, we recommend lowering the batch size by appending ```data_loader.train.batch_size="<batch-size>"``` to the above command. For running the training as a job chain on slurm or resuming training, always set the correct path in ```hydra.run.dir="<runtime-dir>"``` for each task.


## BibTeX
If you are planning to use MV-LDM in your work, consider citing it as follows.
<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <pre><code>@misc{asim24met3r,
    title = {MET3R: Measuring Multi-View Consistency in Generated Images},
    author = {Asim, Mohammad and Wewer, Christopher and Wimmer, Thomas and Schiele, Bernt and Lenssen, Jan Eric},
    booktitle = {arXiv},
    year = {2024},
}</code></pre>
  </div>
</section>


