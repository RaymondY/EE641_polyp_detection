# EE641_polyp_detection

## How to set up the environment
1. Install [Anaconda](https://www.anaconda.com/products/individual)
2. Create a new environment with the following command:
```
conda create --name polyp_detection python=3.10
conda activate polyp_detection

conda install numpy scipy scikit-learn tqdm matplotlib

# for NVIDIA GPU
# conda install -c "nvidia/label/cuda-11.7.0" cuda-toolkit
# modify the cuda version according to your system
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# for Apple
conda install pytorch torchvision -c pytorch

pip install opencv-python
```