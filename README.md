# DCGAN
#tensorflow


the program implement this paper: https://arxiv.org/pdf/1511.06434.pdf

#Prerequisites
python 3.5
tensorflow 0.12.1
gpu or cpu

#Dataset
the dataset is celebA, you can download here: mmlab.ie.cuhk.edu.hk/projects/CelebA.html, and then put your download images into celebA folder

#Usage
To train a model
$ python main.py --train True
To test a model
$ python main.py --train False
