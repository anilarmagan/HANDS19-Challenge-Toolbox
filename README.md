# [A Toolbox for HANDS'19 Challenge](https://sites.google.com/view/hands2019/challenge)

<p align="justify">
This repository is created to provide visualization tools for HANDS'19 Challenge dataset and help the participants getting started. 
The HANDS19 Challenge is a public competition designed for the evaluation of the task of 3D hand pose estimation in both depth and colour modalities in the presence and absence of objects. The main goals of this challenge are to assess the performance of state of the art approaches in terms of interpolation-extrapolation capabilities of hand variations in their main four axes (shapes, articulations, viewpoints, objects), and the use of synthetic data to fill the gaps of current datasets on these axes. Parameters of a fitted hand model (MANO) and a toolkit to synthesize data are provided to participants for synthetic image generation. Training and test splits are carefully designed to study the interpolation and extrapolation capabilities of participants' techniques on these mentioned axes and the potential benefit of using such synthetic data. The challenge consists of a standardized dataset, an evaluation protocol for three different tasks and a public competition.
</p>

An analysis of the participated approaches and the results are summarised in:

[Measuring Generalisation to Unseen Viewpoints, Articulations, Shapes and Objects for 3D Hand Pose Estimation under Hand-Object Interaction](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123680086.pdf)

Video presentations to be presented at [ECCV'20](https://eccv2020.eu/) can be found here:

|         1 minute short video:          |           10 minute long video:          |
|:---:|:---:|
| <a href="https://drive.google.com/file/d/1Gha4KOlNG8qEdpU23yQJTjUVtaLHm_Ev/preview"> <img src="https://drive.google.com/uc?export=view&id=1o3_VPzCvZ-aeXl65uITnvssmXjx3bijc" alt="1min short video" height="258" width="358"> </a> | <a href="https://drive.google.com/file/d/1QlByA9wa5Ty9vAJlUyc9EPbV3Yk6K0J0/preview"> <img src="https://drive.google.com/uc?export=view&id=1yj4SMhL9c6TfyAs4liIn0TDr8MeXaW8_" alt="10min long video" height="248" width="358"> |
	
If you find our work useful in your research, please consider **citing** our paper:
```
@InProceedings{armaganeccv2020,
 title   = {Measuring Generalisation to Unseen Viewpoints, Articulations, Shapes and Objects for 3D Hand Pose Estimation under Hand-Object Interaction},
 author  = {Anil Armagan and Guillermo Garcia-Hernando and Seungryul Baek and Shreyas Hampali and Mahdi Rad and Zhaohui Zhang and Shipeng Xie and MingXiu Chen and Boshen Zhang and Fu Xiong and Yang Xiao and Zhiguo Cao and Junsong Yuan and Pengfei Ren and Weiting Huang and Haifeng Sun and Marek Hr\'{u}z and Jakub Kanis and Zden\v{e}k Kr\v{n}oul and Qingfu Wan and Shile Li and Linlin Yang and Dongheui Lee and Angela Yao and Weiguo Zhou and Sijia Mei and Yunhui Liu and Adrian Spurr and Umar Iqbal and Pavlo Molchanov and Philippe Weinzaepfel and Romain Br\'{e}gier and Gr\'{e}gory Rogez and Vincent Lepetit and Tae-Kyun Kim},
 booktitle = {European Conference on Computer Vision ({ECCV})},
 year = {2020}
}
```

### Getting Started
- Download the challenge dataset. HANDS'19 is hosted on [CodaLab](https://competitions.codalab.org/competitions/20913#learn_the_details) for you to signin and participate. Please fill the corresponding form on CodaLab to get access to the dataset. Please follow the [challenge website](https://sites.google.com/view/hands2019/challenge) for more details on the dataset.
- The toolbox should be under the same directory as HANDS'19 dataset.
- Download the MANO right hand model from [MANO download page](http://mano.is.tue.mpg.de) and place it under the toolbox directory.
- Dataset images with ground-truth annotations and renderings of the MANO model can visualized with **visualize_task1_task2.py** for Task 1 and Task 2, and with **visualize_task3.py** for Task 3.

```
usage:  cd your_directory/HANDS19-Challenge-Toolbox

	python3 visualize_task1_task2.py --task-id=1 --frame-id=0 --mano-model-path=./MANO_RIGHT.pkl
	OR
	python3 visualize_task3.py --frame-id=0 --mano-model-path=./MANO_RIGHT.pkl
```

### Dependencies
This code is tested under Ubuntu16.04 with python3.5.

Python dependencies can be installed with **`pip3 install -r requirements.txt`**. Some dependencies are listed as: torch, moderngl, opendr, opencv-python, argparse, pickle, numpy and matplotlib. Please see `requirements.txt` for version specifications. 

### Licence
<p align="justify">
The download and use of the datasets and the software released for academic research only. It is free to use for researchers from educational or research institutes for non-commercial purposes. When downloading the challenge dataset you agree to (unless with expressed permission of the authors): not redistribute, modificate, or commercial usage of this dataset in any way or form, either partially or entirely.
</p>

## Acknowledgements
This code has been written and modified by Anil Armagan and Seungryul Baek using the sources below and Shreyas Hampali's sources on visualization of [HO-3D dataset](https://arxiv.org/abs/1907.01481v1). 

Thanks to the below sources we are able to provide this code:
- [MANO model](https://www.is.mpg.de/uploads_file/attachment/attachment/392/Embodied_Hands_SiggraphAsia2017.pdf) is licensed under: http://mano.is.tue.mpg.de/license 
- The provided MANO class is adapted from [PyTorch implementation](https://github.com/MandyMo/pytorch_HMR/blob/master/src/SMPL.py) of a great work by [Kanazawa et al.](https://arxiv.org/pdf/1712.06584.pdf).
- Rendering codes for Task 3 MANO model is implemented with [OpenDr](https://github.com/mattloper/opendr).

```
@article{MANO:SIGGRAPHASIA:2017,
      title = {Embodied Hands: Modeling and Capturing Hands and Bodies Together},
      author = {Romero, Javier and Tzionas, Dimitrios and Black, Michael J.},
      journal = {ACM Transactions on Graphics, (Proc. SIGGRAPH Asia)},
      volume = {36},
      number = {6},
      series = {245:1--245:17},
      month = nov,
      year = {2017},
      month_numeric = {11}
}
@article{hampali2019arxivv1,
	author = {{Hampali}, Shreyas and {Oberweger}, Markus and {Rad}, Mahdi and {Lepetit}, Vincent},
	title = "{HO-3D: A Multi-User, Multi-Object Dataset for Joint 3D Hand-Object Pose Estimation}",
	journal = {arXiv e-print arXiv:1907.01481v1},
	year = {2019}
}
@InProceedings{sbaekcvpr2019,
	title={Pushing the envelope for RGB-based dense 3D hand pose estimation via neural rendering},
	author={Seungryul Baek and Kwang In Kim and Tae-Kyun Kim},
	journal={Proceedings of the {IEEE} Conference on Computer Vision and Pattern Recognition ({CVPR})},
	year={2019}
}
@InProceedings{hmrKanazawa17,
  title={End-to-end Recovery of Human Shape and Pose},
  author = {Angjoo Kanazawa and Michael J. Black and David W. Jacobs and Jitendra Malik},
  booktitle={Computer Vision and Pattern Regognition (CVPR)},
  year={2018}
}
```
