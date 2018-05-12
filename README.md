# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, we develop code for an image classifier built with PyTorch, then convert it into a command line application.

## Requirements-

* Python 3.6+
* Pytorch 0.4.0+

# Running-

## Dataset-

* The dataset is available at - https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz
* Uncompress it in the __data__ directory.
* Or use this command to set it up on linux-
```
mkdir data && cd data && wget https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz && tar -xvzf flower_data.tar.gz
```

## Notebook and scripts-

* Use __Image Classifier Project.ipynb__ to train and test a model based on resnet152 architecture.
* Use __train.py__ to train the model on __resnet152__ or __vgg19_bn__ architecture.
* run `python train.py -h` to see all the command line options.
* Sample command- `python train.py --gpu --batch_size=128 --save_path=checkpoint.model --arch=vgg19_bn --epochs=5`
* Use __predict.py__ to make predictions on an image.
* run `python predict.py -h` to see all the command line options.
* Sample command- `python predict.py checkpoint.model data/test/99/image_07833.jpg --gpu --topk=5`
