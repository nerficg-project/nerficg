<!-- <img src='resources/demo_llff.gif' width=320 height=180><img src='resources/demo_synthetic.gif' width=320 height=180> -->
<img src='resources/nerficg_banner.png' width=100%>

<!-- ---
# NeRFICG -->

![Python](https://img.shields.io/static/v1?label=Python&message=3.11&color=success&logo=Python)&nbsp;![PyTorch](https://img.shields.io/static/v1?label=Pytorch&message=2.5&color=success&logo=PyTorch)&nbsp;![CUDA](https://img.shields.io/static/v1?label=CUDA&message=11.8&color=success&logo=NVIDIA)&nbsp;![OS](https://img.shields.io/static/v1?label=OS&message=Linux&color=success&logo=Linux)&nbsp;[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](https://opensource.org/licenses/MIT)

A flexible Pytorch framework for simple and efficient implementation of neural radiance fields and rasterization-based view synthesis methods, including a GUI for interactive rendering.

<div class="image-row">
<a href="resources/teaser_videos/inpc.mp4">
	<img src="resources/teaser_videos/inpc.gif" width="48%"/>
</a>
<a href="resources/teaser_videos/dnpc.mp4">
	<img src="resources/teaser_videos/dnpc.gif" width="48%"/>
</a>
</div>


## Getting Started

- This repository uses submodules, clone using one of the following options:
	```shell
	# HTTPS
	git clone https://github.com/nerficg-project/nerficg.git --recursive && cd nerficg
	```
	or
	```shell
	# SSH
	git clone git@github.com:nerficg-project/nerficg.git --recursive && cd nerficg
	```

- Install the global dependencies listed in `scripts/condaEnv.sh`, or automatically create a new conda environment by executing the script:
	```shell
	./scripts/condaEnv.sh && conda activate nerficg
	```

- To install all additional dependencies for a specific method, run:
	```shell
	./scripts/install.py -m <METHOD_NAME>
	```
	or use
	```shell
	./scripts/install.py -e <PATH_TO_EXTENSION>
	```
	to only install a specific extension.

- [optional] To use our training visualizations with [Weights & Biases](https://wandb.ai/site), run the following command and enter your account identifier:
	```shell
	wandb login
	```


## Creating a Configuration File

To create a configuration file for training, run
```
./scripts/defaultConfig.py -m <METHOD_NAME> -d <DATASET_TYPE> -o <CONFIG_NAME>
```
where `<METHOD_NAME>` and `<DATASET_TYPE>` match one of the items in the `src/Methods` and `src/Datasets` directories, respectively.
The resulting configuration file `<CONFIG_NAME>.yaml` will be available in the `configs` directory and can be customized as needed.
To create a directory of configuration files for all scenes of a dataset, use the `-a` flag. This requires the full dataset to be available in the `dataset` directory.


## Training a New Model

To train a new model from a configuration file, run:
```
./scripts/train.py -c configs/<CONFIG_NAME>.yaml
```
The resulting images and model checkpoints will be saved to the `output` directory.

To train multiple models from a directory or list of configuration files, use the `scripts/sequentialTrain.py` script with the `-d` or `-c` flag respectively.


## Training on custom image sequences

If you want to train on a custom image sequence, create a new directory with an `images` subdirectory containing all training images.
Then you can prepare the image sequence using the provided [COLMAP](https://colmap.github.io) script, including various preprocessing options like monocular depth estimation, image segmentation and optical flow.
Run
```
./scripts/colmap.py -h
```
to see all available flags and options. Alternatively, you can try to use the `scripts/vggsfm.py` script if COLMAP’s SfM pipeline fails.

After calibration, the custom dataset can be loaded by setting `Colmap` or `VGGSfM` as `GLOBAL.DATASET_TYPE` in the config file and entering the correct directory path in the config file under `DATASET.PATH`.


## Inference and evaluation

We provide multiple scripts for easy model inference and performance evaluation after training.
Use the `scripts/inference.py` script to render output images for individual subsets (train/test/eval) or custom camera trajectories defined in `src/Visual/Trajectories` using the `-s` option.
Additional rendering performance benchmarking and metric calculation is available using the `-b` and `-m` flags respectively .

The `scripts/generateTables.py` script further enables consistent metric calculation over multiple pre-generated output image directories (e.g. to compare multiple methods against GT), and automatically generates LaTeX code for tables containing the resulting values.
Use `-h` to see the available options for all scripts.


## Graphical User Interface

To inspect a pretrained model in our GUI, make sure the GUI submodule is initialized, run
```
./scripts/gui.py
```
and select the generated output directory.

Some methods support live GUI interaction during optimization. To enable live GUI support, activate the `TRAINING.GUI.ACTIVATE` flag in your config file.


## Frequently Asked Questions (FAQ)

__Q:__ What coordinate system does the framework use internally?

__A:__ The framework uses a left-handed coordinate system for all internal calculations:
- World space: x-right, y-forward, z-down (left-handed)  
- Camera space: x-right, y-down, z-backward (left-handed)

<!-- use the following template for new questions (remove comment once second question is added)
##
__Q:__

__A:__

-->


## Acknowledgments

We started working on this project in 2021. Over the years many projects have inspired and helped us to develop this framework.
Apart from any reference you might find in our source code, we would specifically like to thank all authors of the following projects for their great work: 
- [NeRF: Neural Radiance Fields](https://github.com/bmild/nerf)
- [NeRF-pytorch](https://github.com/yenchenlin/nerf-pytorch)
- [Instant Neural Graphics Primitives](https://github.com/NVlabs/instant-ngp)
- [ngp_pl](https://github.com/kwea123/ngp_pl.git)
- [MultiNeRF: A Code Release for Mip-NeRF 360, Ref-NeRF, and RawNeRF](https://github.com/google-research/multinerf)
- [torch_efficient_distloss](https://github.com/sunset1995/torch_efficient_distloss)
- [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://github.com/graphdeco-inria/gaussian-splatting)
- [CamP Zip-NeRF: A Code Release for CamP and Zip-NeRF](https://github.com/jonbarron/camp_zipnerf/)
- [ADOP: Approximate Differentiable One-Pixel Point Rendering](https://github.com/darglein/ADOP)


## License and Citation

This framework is licensed under the MIT license (see [LICENSE](LICENSE)).

If you use it in your research projects, please consider a citation:
```bibtex
@software{nerficg,
	author = {Kappel, Moritz and Hahlbohm, Florian and Scholz, Timon},
	license = {MIT},
	month = {1},
	title = {NeRFICG},
	url = {https://github.com/nerficg-project},
	version = {1.0},
	year = {2025}
}
```