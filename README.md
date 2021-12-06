## Acute Lymphoblastic Leukemia Classification Project

In this project, we experiment with several convolutional neural networks (CNNs) in order to find the model that best distinguishes between normal and leukemia blast (cancer) cells. Building a model to accurately classify which type of cell is shown to the computer is extremely important when it comes to diagnosing patients correctly in the medical field.
 
 An overarching interest in becoming more familiar with CNN networks for image classification and applying machine learning skills to a healthcare-related question was the main motivation for choosing this application. The dataset used to answer this question consists of 15,135 images from 118 patients with two classes: Normal cell and Leukemia blast (cancer) cell. The number of images translates to over 10 GB of data.

PyTorch, a machine learning framework, is used to implement the networks built since there are pretrained networks that we will use as a baseline comparison. F-1 Score and Cohen will be used to measure the performance of the network.
Models experimented on this dataset include VGGNet16, VGGNet19, EfficientNetB7, EfficientnetB3, EfficientnetB4, EfficientnetB5, EfficientnetB0, GoogleNet, Densenet161, DenseNet121, DenseNet169, and ResNet152.



##

The code for the best performing models on this project (obtaining ~77% accuracy and ~77% weighted F1-score) are included in the Code folder of the main branch. Codes for all experiements by respective team members are included in separate branches.

The dataset used for this project was obtained [through Kaggle](https://www.kaggle.com/andrewmvd/leukemia-classification), which makes available the second version of this dataset. 

##### Data Citation

Gupta, A., & Gupta, R. (2019). ALL Challenge dataset of ISBI 2019 [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/tcia.2019.dc64i46r
