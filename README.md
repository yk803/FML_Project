# HAAT: Improve Adversarial Robustness via Hessian Approximation

This repository is the final project of NYU CSCI-GA 2566 [Foundations of Machine Learning](https://cs.nyu.edu/~mohri/mlsp22/) at Spring 2022. This project is the work of [Panchi Mei](https://github.com/Panchi-Mei), [Yukai Yang](https://github.com/yk803), and [Zequan Wu](https://github.com/zw2700).

# Abstract

# How to Run the Code

## Train The Model
To completely rerun the whole experiment, one can download this repository by
```
git clone https://github.com/yk803/FML_Project.git
```

Next, to train the model, if you have access to the NYU HPC server or a similar one that can run slurm script, one can revise the account and file directory to the current one, and call
```
sbatch at_train_base.slurm
sbatch at_train.slurm
```
to run the training task.

If HPC is not available, one can, in this directory, run
```
python train_base.py checkpoint_base 10
python train.py checkpoint_soar 10 1e-6 1e-1 5
```

This will run the same training task locally.

The first line is for base PGD training. It requires two inputs: the name of the directory where models are stored, and an integer **X** meaning we make a checkpoint every **X** epochs in training.

The second line is for our Second-order approximation training. It has 3 additional inputs: Xi, the step size for doing finite difference approximation; learning rate for the whole model, and step_fd, the number of steps for finite difference approximation.


## Test The Model
The testing file uses solely [AutoAttack](https://github.com/fra31/auto-attack), as required in the course project. 

It can be executed by
```
sbatch at_test_base.slurm
sbatch at_test.slurm
```
on NYU HPC.

If that is not available, just run
```
python test.py best_pgd_adversarial_training.pt checkpoint_base
python test.py best_pgd_adversarial_training.pt checkpoint_soar
```
This file has two inputs: filename of the checkpoint, and the directory it is stored.



# Acknowledgement
* The baseline code for ResNet and the vanilla PGD training is adapted from the work of [Dongbig Na](https://github.com/ndb796/Pytorch-Adversarial-Training-CIFAR).
