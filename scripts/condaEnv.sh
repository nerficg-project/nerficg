#! /usr/bin/env bash

ENV_NAME="nerficg"
PYTHONVERSION="3.11"
CUDAVERSION="cu118"
HEADLESS=false

# parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--name)
            ENV_NAME="$2"
            shift 2
            ;;
        -p|--python)
            PYTHONVERSION="$2"
            shift 2
            ;;
        -c|--cuda)
            CUDAVERSION="$2"
            shift 2
            ;;
        -h|--headless)
            HEADLESS=true
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [-n|--name <name>] [-p|--python <version>] [-c|--cuda <version>] [-h|--headless]"
            exit 1
            ;;
    esac
done

echo "Creating conda environment '$ENV_NAME' with Python $PYTHONVERSION and CUDA $CUDAVERSION"
if [ "$HEADLESS" = true ]; then
    echo "Installing in headless mode (no GUI dependencies)"
fi

# create new conda environment
conda create -y --name $ENV_NAME python=$PYTHONVERSION
# install base dependencies
conda install -y -n $ENV_NAME packaging
conda run -n $ENV_NAME pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/${CUDAVERSION}
conda run -n $ENV_NAME pip install numpy tqdm natsort GitPython av ffmpeg-python pyyaml munch tabulate wandb opencv-python kornia torchmetrics lpips einops setuptools plyfile matplotlib timm plotly pillow jax
conda install -y -n $ENV_NAME -c conda-forge colmap
# install gui dependencies
if [ "$HEADLESS" = false ]; then
    conda run -n $ENV_NAME pip install imgui[sdl2] cuda-python platformdirs
fi
