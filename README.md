# Indian Food Image Classifier (EfficientNet-B0)

A deep learning model that classifies Indian food images into 5 categories using transfer learning with EfficientNet-B0.

I built this to practice transfer learning after finishing the CNN basics of my deep learning course. Instead of training a CNN from scratch (which needs a lot of data and compute), I used a pretrained EfficientNet-B0 and only trained a new classifier head on top of it.

## Classes

- biryani
- kachori
- jalebi
- poha
- aloo_tikki

## Dataset

[Indian Food Images Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/indian-food-images-dataset) from Kaggle. The full dataset has ~80 classes, I picked 5 for this project and split them 80/20 into train/validation.

## Approach

- Loaded EfficientNet-B0 pretrained on ImageNet
- Froze all the convolution/feature extraction layers
- Replaced the final classifier layer with a new one for 5 classes
- Trained only the classifier layer (Adam, CrossEntropyLoss, 10 epochs, batch size 32)
- Used basic augmentation (horizontal flip, rotation) on training data only

I went with freezing the backbone instead of fine-tuning the whole network since the dataset is small and this is closer to what I've actually studied so far.

## Results

- Validation accuracy: ~XX% (fill in after your run)
- Training/validation loss plotted over epochs, best model saved based on validation accuracy

## Files

- `Indian_Food_Classifier_EfficientNetB0.ipynb` — full notebook (data prep, training, evaluation, inference)
- `food_classifier.pth` — saved model weights

## What I'd try next

- Add more classes from the full dataset
- Try unfreezing the last few layers for fine-tuning
- Deploy it as a small Gradio demo

## Tech

Python, PyTorch, Torchvision, Google Colab (GPU)
