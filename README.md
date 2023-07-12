# Bee Wings Classifier

This repository contains an image classification model trained on a collection of wing images for the conservation of honey bees (Apis mellifera) biodiversity in Europe. The model is built with Fastai's vision API using the ResNet152 architecture and a progressive image resize strategy.

The dataset used for training can be found [here](https://huggingface.co/datasets/smaciu/bee-wings-large).

The model's goal is to classify bee wings, which can assist in studying and preserving the biodiversity of honey bees in Europe.

## Technical Details

The model was trained using the Fastai API. Progressive image resize was employed throughout the training to enhance the model's performance. The model architecture used was ResNet152.

## Dataset Normalization

The dataset's mean and standard deviation are used to normalize the images:

```python
bee_wing_stats =([0.7641, 0.7641, 0.7641], [0.1771, 0.1771, 0.1771])
```

## Model Training

Progressive resizing was applied throughout the training process. The details of the training phases are outlined in the `prog_list` dictionary in the source code.

## Dependencies

To run the scripts in this repository, the following libraries need to be installed:

- PyTorch
- Fastai
- huggingface_hub

## Installing and Running

To clone and run this application, you'll need [Git](https://git-scm.com) and [Python](https://www.python.org/downloads/) installed on your computer. From your command line:

```
$ git clone https://github.com/smaciu/bee-wings.git
$ cd bee-wings
$ pip install -r requirements.txt
$ python predict.py path_to_your_image.jpg
```

To predict multiple images from folder:

**prediction.csv** will be saved in the main repo folder with original image name, predicted country of orign and probability

```
$ git clone https://github.com/smaciu/bee-wings.git
$ cd bee-wings
$ pip install -r requirements.txt
$ python batch_predict.py path_to_your_folder_with_images

```

## Contributors

Slawek Maciura 
email: slamaciu@icloud.com

## License

This project is licensed under the terms of the MIT license.
