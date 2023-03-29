# DBlur: An Image Deblurring Toolkit

## Table of Contents

1. [About](#1-about)
2. [Installation](#2-installation)
3. [Supported Models](#3-supported-models)
4. [Datasets](#4-datasets)
5. [Usage](#5-usage)
    - [Training, testing, deblurring with default settings](#51-training-testing-deblurring-with-default-settings)
    - [Customize training pipeline for a model](#52-customize-training-pipeline-for-a-model)
    - [Customize testing pipeline for a model](#53-customize-testing-pipeline-for-a-model)
    - [Deblur images with multiple pretrained models](#54-deblur-images-with-multiple-pretrained-models)
6. [Call for Contributions](#6-call-for-contributions)
7. [Citation](#7-citation)

## 1. About

[DBlur](https://pypi.org/project/dblur/) is an open-source python library for image deblurring. It is simple and highly 
versatile making it perfect for both experts and non-experts in the field. 

For example, training, testing and deblurring with numerous SOTA models can be performed with just 2-3 lines of code 
with the default parameters mentioned in each paper. On the other hand, these methods can abe highly customised to 
assist existing researchers explore new avenues in the filed. For more details regarding its usage, refer to 'Usage' 
section below. 

DBlur has a wide range functionalities which include:
1. Numerous highly customisable SOTA model architectures for image deblurring. The list of supported models are 
mentioned [here](#3-supported-models).
2. Numerous popular and novel datasets which can be found 
[here](https://www.kaggle.com/datasets/jishnuparayilshibu/a-curated-list-of-image-deblurring-datasets).
3. Ready-to-use pipelines created according to the paper for training each of these models on a given dataset.
4. Ready-to-use pipelines for testing each of these models.
5. Multiple post-processing steps to improve deblurring performance. 
6. Deblurring images using pretrained models.
7. Deblurring images by combining multiple pretrained models in different ways.
8. Common evaluations metrics such as PSNR, SSIM, MSE and Brisque. 

## 2. Installation
Install [Dblur](https://pypi.org/project/dblur/) with pip:

```bash
$ pip install dblur
```

The following requirements shall be installed alongside Dblur:

* torch 
* scipy
* numpy
* torchmetrics
* tensorboard
* piq

## 3. Supported Models
1. [TextDBN](http://www.bmva.org/bmvc/2015/papers/paper006/paper006.pdf)
2. [MSCNN](https://arxiv.org/abs/1702.02359)
3. [SRN](https://arxiv.org/abs/1802.01770)
4. [StackDMPHN](https://arxiv.org/abs/1904.03468)
5. [Restormer](https://arxiv.org/abs/2111.09881)
6. [NAFNet](https://arxiv.org/abs/2204.04676)
7. [FNAFNet](https://arxiv.org/abs/2111.11745)

## 4. Datasets

A curated list of popular and custom datasets for general, face, and text deblurring can be found 
[here](https://www.kaggle.com/datasets/jishnuparayilshibu/a-curated-list-of-image-deblurring-datasets). This datasets 
have been preprocessed/structured so that they can be easily used with our library. 

## 5. Usage
Before diving into the usage of the library, it is important to look at a brief overview of the structure of the 
library. The two main classes associated with each deblurring model is a respective Trainer and Tester class. These 
classes for each model are all subclassed from a main BaseTrainer and BaseTester class. 

Broadly speaking, each Trainer class has the following methods:
* train()
* validate()
* get_model()
* get_loss()
* get_optimizer()
* get_lr_scheduler()
* get_train_dataloader()
* get_val_dataloader()

Similarly, each Tester class has the following methods:
* test()
* deblur_imgs()
* deblur_single_img()
* get_test_dataloader()


All methods which are part of the public API have an elaborate docstring. Hence, usage of the library is made 
relatively straightforward. 
Following is a brief description on the usage of the library.

### 5.1. Training, testing, deblurring with default settings.

This section is mainly aimed at non-experts who would like to train, test a model or deblur images with a pretrained 
model with default settings provided in each paper. By default settings, we mean:

* optimizer 
* learning rate scheduler 
* loss function
* parameters associated with the model architecture, loss, scheduler, optimizer and others that are used for 
* training/testing (e.g. batch size).

All the components metioned above are part of the setting. Below, we demonstrate how to train, test and deblur using 
the default settings for the Restormer model (can be easily generalised to other models).

```python
from dblur.default.restormer import train, test, deblur_imgs, deblur_single_img

train_img_dir = "path_to_training_dataset"
val_img_dir = "path_to_validation_dataset"
test_img_dir = "path_to_test_dataset"
model_path = "path_for_model"

# Train model
train(model_path, train_img_dir)

# Test pretrained model
train(model_path, test_img_dir)

# Deblur images in a directory using pretrained model
deblur_imgs(model_path, "blurred_imgs_path", "sharp_imgs_path")

# Deblur single image using pretrained model
deblur_single_img(model_path, "blurred_img_path", "sharp_img_path")
```

### 5.2. Customize training pipeline for a model

The following code illustrates the entire training pipeline for the Restormer (can be generalised to any other model).

```python
from dblur.trainers.restormer import RestormerTrainer

restormer_trainer = RestormerTrainer()
train_dataloader = restormer_trainer.get_train_dataloader(train_img_dir, batch_size=8)
val_dataloader = restormer_trainer.get_val_dataloader(val_img_dir, batch_size=8)
model = restormer_trainer.get_model(num_layers=4, num_refinement_blocks = 2)
optimizer = restormer_trainer.get_optimizer(model.parameters())
loss_func = restormer_trainer.get_loss()
lr_scheduler = restormer_trainer.get_lr_scheduler(optimizer)

restormer_trainer.train(model,
                        train_dataloader,
                        val_dataloader,
                        optimizer,
                        loss_func,
                        save_checkpoint_freq=10,
                        logs_folder='runs',
                        checkpoint_save_name="path_to_model",
                        val_freq=100, 
                        write_logs=True,
                        lr_scheduler=lr_scheduler,
                        epochs=epochs)
```

Each method of the RestormerTrainer object has multiple parameters that can be set accordingly. This can be found in 
the docstring of each method. The example above only illustrates few of the parameters that can be specified. Moreover, 
the user is not limited to using the loss function or optimizer given by the methods of the RestormerTrainer object.
 
### 5.3. Customize testing pipeline for a model

The following code illustrates the entire testing pipeline for the Restormer (can be generalised to any other model).

```python
from dblur.testers.restormer import RestormerTester

restormer_tester = RestormerTester()
test_dataloader = restormer_tester.get_test_dataloader(test_img_dir, batch_size=8)
model = restormer_tester.get_model(num_layers = 4, num_refinement_blocks = 2)
loss_func = restormer_tester.get_loss()

# test model on test dataset.
restormer_tester.test(model,
                      model_path,
                      test_dataloader,
                      loss_func,
                      is_checkpoint=True,
                      window_slicing=True,
                      window_size=256)
                      
# deblur images in a directory using pretrained model
restormer_tester.deblur_imgs(model,
                             model_path,
                             blur_img_dir,
                             sharp_img_dir,
                             is_checkpoint=True,
                             batch_size=8,
                             window_slicing=False)

#deblur single image using pretrained model
restormer_tester.deblur_single_img(model,
                                   model_path,
                                   blur_img_path,
                                   sharp_img_path,
                                   is_checkpoint=True,
                                   window_slicing=False)
```

Each method of the RestormerTester object has multiple parameters that can be set accordingly. This can be found in 
the docstring of each method. The example above only illustrates few of the parameters that can be specified. 

### 5.4. Deblur images with multiple pretrained models

Dblur provides two ways in which you can combine multiple pretrained models to deblur images.

a) First method averages the outputs of the pretrained models provided. The code below shows how to perform this on a 
single image (can be done for multiple images in a directory as well).

```python
from dblur.testers.mscnn import MSCNNTester
from dblur.testers.stack_dmphn import StackDMPHNTester
from dblur.multi_modal_deblur import multi_modal_deblur

mscnn_tester = MSCNNTester()
model1 = mscnn_tester.get_model()

dmphn_tester = StackDMPHNTester()
model2 = dmphn_tester.get_model(num_of_stacks=1)

multi_modal_deblur(models=[model1, model2], 
                   model_names=["MSCNN", "StackDMPHN"],
                   model_paths=[model_path1, model_path2],
                   blur_img_path=blur_img_path,
                   sharp_img_path=sharp_img_path,
                   is_checkpoint=[True, True], 
                   window_slicing=True, 
                   window_size=256, 
                   overlap_size=0)
```

b) The second method selects the output with the lowest(i.e. the best) Brisque index. Brisque is an index used to access
the quality of an image. For more details regarding Brisque, refer to 
[this](https://live.ece.utexas.edu/publications/2012/TIP%20BRISQUE.pdf). The code below shows how to perform this on a 
single image (can be done for multiple images in a directory as well).

```python
from dblur.testers.mscnn import MSCNNTester
from dblur.testers.stack_dmphn import StackDMPHNTester
from dblur.multi_modal_deblur import multi_modal_deblur_by_brisque

mscnn_tester = MSCNNTester()
model1 = mscnn_tester.get_model()

dmphn_tester = StackDMPHNTester()
model2 = dmphn_tester.get_model(num_of_stacks=1)

multi_modal_deblur_by_brsique(models=[model1, model2], 
                              model_names=["MSCNN", "StackDMPHN"],
                              model_paths=[model_path1, model_path2],
                              blur_img_path=blur_img_path,
                              sharp_img_path=sharp_img_path,
                              is_checkpoint=[True, True], 
                              window_slicing=True, 
                              window_size=256, 
                              overlap_size=0)
```

## 6. Call for Contributions

Dblur is a project which is in active development. Any helpful comments and improvements are highly 
encouraged. To do so, please open an issue in this Github page. 

In particular, due to the lack of GPUs on our side, the pretrained models for each of the model architectures are not 
yet available. Hence, if you have trained any of the models on a particular dataset, you are highly encouraged to share
this pretrained model on our Github page. This way, the pretrained models can be added to the library. 

## 7. Citation
If you use or extend this work, please consider citing it as below:

```
@software{Parayil_Shibu_DBlur_An_Image_2023,
author = {Parayil Shibu, Jishnu},
month = {3},
title = {{DBlur: An Image Deblurring Toolkit}},
url = {https://github.com/Jishnu8/DBlur-An-Image-Deblurring-Toolkit},
version = {1.0.0},
year = {2023}
}
```
