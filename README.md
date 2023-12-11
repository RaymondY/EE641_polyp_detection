# EE641_polyp_detection

## Environment
### Requirements
* Python 3.10.13
* PyTorch 2.1.1
* torchvision 0.16.1
* numpy 1.26.2
* scikit-learn 1.3.0
* tqdm 4.65.0
* matplotlib 3.8.0
* opencv-python 4.8.1.78

### How to set up the environment
1. Install [Anaconda](https://www.anaconda.com/products/individual)
2. Create a new environment with the following command:
```
conda create --name polyp_detection python=3.10
conda activate polyp_detection

conda install numpy scikit-learn tqdm matplotlib

# for NVIDIA GPU
# conda install -c "nvidia/label/cuda-11.7.0" cuda-toolkit
# modify the cuda version according to your system
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# for Apple
conda install pytorch::pytorch torchvision -c pytorch

pip install opencv-python
```

## How to run the code
1. Set up the environment as described above.
2. Download the dataset from [Kvasir-SEG](https://datasets.simula.no/downloads/kvasir-seg.zip), then unzip it to the "datasets" folder.
3. Run the following command to preprocess the dataset:`python preprocess.py` if this is the first time to run the code.
4. Run the following command to train the model: `python main.py`. Comment the lines in run() function under _"test example"_ in main.py to evaluate the saved model.


## Files
* `preprocess.py`: preprocess the dataset (resize)
* `main.py`: main file to run the code
* `model.py`: model definition
* `loss.py`: loss function
* `utils.py`: utility functions
* `config.py`: configuration file
* `train.py`: training function
* `evaluate.py`: evaluation function
