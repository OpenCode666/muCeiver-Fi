# μCeiver-Fi: Exploiting Spectrum Resources of Multi-link Receiver for Fine-Granularity Wi-Fi Sensing

Welcome to μCeiver-Fi. This repository contains the resource code to implement μCeiver-Fi system.


## Contents
[Introduction](#introduction)

[Getting Started](#getting-started)

[Workflow](#workflow)

[Evaluation](#evaluation)


## Introduction

μCeiver-Fi is a framework that relies solely on a commodity multi-link receiver to leverage spectrum resources for Wi-Fi 3D human pose estimation.


## Getting Started

### Environment and Hardware (optional)
Ubuntu 22.04.3 LTS

6.8.0-60-generic kernel

Python 3.8.16

CUDA Version: 12.2.

NVIDIA RTX A5000

MATLAB R2023b


### Install

1. Download the [Source Code](https://github.com/OpenCode666/muCeiver-Fi) and [Dataset](https://zenodo.org/records/16209360).

2. Install Python: Please ensure Python 3.8.16 is installed on your computer. You can also download the Python source from the [official website](https://www.python.org/).

3. Set Up Virtual Environment: It is recommended to set up a virtual environment to ensure a clean and isolated environment for μCeiver-Fi implementation. Tools like **conda** can be used for this purpose. Make sure to activate your virtual environment before proceeding.

4. Install the necessary packages: We provide the requirements.txt in source code. You can install them by ```pip install -r requirements.txt```.



## Workflow

1. Train the human pose estimation neural network:

- Open ```mian.py``` in the source code, and change the path in line 21 to the folder path of the Dataset. This folder contains multi-link receiver signals and their corresponding labels.

- Run ```main.py```, which will create the directory ```./experiments```, then save the prediction results as ```./deterministic/training/predictions/pre_result.mat``` and save the model as ```./deterministic/checkpoints/model_epochn.pth```.

- File ```args_det.py``` contains the neural network's training parameters. Based on our tests, the current parameters achieve satisfactory results. Users can modify them as needed, since our model demonstrates strong robustness.

2. Post-processing:

- Transfer the generated ```./experiments``` folder to the machine used for running MATLAB.

- Open ```post_proc``` $\rightarrow$ ```post_proc.m``` in the source code, and change the path in line 11 to the folder path of the ```experiments```.

- Run ```post_proc.m```, which will print the Keypoint Average Localization Error in the MATLAB command window and save the post-processed result ```HPE.mat``` in the current directory.

- Run ```visualization.m```, which will visually present six poses: ```Walking```, ```Pointing```, ```Hands up```, ```Hands open```, ```Sitting down```, and ```Standing```. In addition, we provide the specific poses corresponding to each test sample in ```real_pose``` $\rightarrow$ ```real_pose.mat```. Users can adjust the index in the ```visualization.m``` to check whether the predicted results successfully achieve pose estimation.
        



## Evaluation

1. Description

- ```pre_result.mat```: Neural network model’s prediction results.

- ```model_epoch.pth```: Neural network model.

- ```Keypoint Average Localization Error```: Printed in the MATLAB command window.

- ```HPE.mat```: Human pose estimation results.

- ```Visualizations```: Visualizations of human pose estimation.



2. Expected results

- The Keypoint Average Localization Error is approximately 5 cm, and the visualizations of six typical poses, including ```Walking```, ```Pointing```, ```Hands up```, ```Hands open```, ```Sitting down```, and ```Standing```, are accurately rendered, which together indicate minimal distortion in human pose estimation and demonstrate satisfactory performance of our μCeiver-Fi.
Furthermore, considering that the typical distance between the head and neck is about 15 cm, the system’s ability to accurately distinguish these keypoints suggests that our μCeiver-Fi achieves GHz-level range resolution.
