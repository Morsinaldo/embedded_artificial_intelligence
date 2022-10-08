# Model Card
Model cards are a succinct approach for documenting the creation, use, and shortcomings of a model. The idea is to write a documentation such that a non-expert can understand the model card's contents. For additional information see the Model Card paper: [![Arxiv](https://img.shields.io/badge/paper-arxiv-red)](https://arxiv.org/pdf/1810.03993.pdf)

## Model Details
Morsinaldo Medeiros created the model. A complete data pipeline was built using Google Colab, Scikit-Learn and Weights & Bias to train a MLP (Multilayer Perceptron) model. The big-picture of the data pipeline is shown below:

<center><img width="800" src="./images/workflow.png"></center>

Model version 0 was trained with two hidden layers with 100 and 20 neurons, respective. After hyperparameter tuning, the best configuration founded is: two hidden layers with 100 neurons each, relu activation function, Adam solver, alpha = 0.0001 and constant learning rate.

## Intended Use

This model is used as a proof of concept for the evaluation of an entire data pipeline incorporating Machine Learning fundamentals and as a comparison with KNN algorithm. The KNN evaluation on this dataset was made by [Ivanovitch Silva](https://github.com/ivanovitchm/embedded.ai/tree/main/lessons/week_05/first_classifier#training-data). The data pipeline is composed of the following stages: 1 - `fecht data`, 2 - `preprocess`, 3 - <s>`check data`</s>, 4 - `segregate`, 5 - `train` and 6 - `test`. For simplify, we remove the data checking from this pipeline.

## Training Data

The purpose of this dataset is to correctly classify an image as containing a dog, cat, or panda. Containing only 3,000 images, the Animals dataset is meant to be another introductory dataset that we can quickly train a KNN model or a MLP modelo and obtain initial results (no so good accuracy) that has potential to be used as a baseline. In future works, we intend compare the metrics from these two algorithms with CNN (Convolutional Neural Network) metrics on this dataset.

After the EDA stage of the data pipeline, it was noted that the images in training data has different resolutions. A pre-processing stage is necessary in order to normalize all images using the same size.

<center><img width="800" src="./images/EDA.png"></center>

## Evaluation Data

The dataset under study is split into Train and Test during the `Segregate` stage of the data pipeline. 75% of the clean data is used to Train and the remaining 25% to Test. For hyperparameter tuning, we get Train dataset  and split again into a new Train and Validation with 75% for Train and 25% remaining for Validation.

## Metrics
In order to follow the performance of machine learning experiments, the project marked certains stage outputs of the data pipeline as metrics. The metrics adopted are: [accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html), [f1](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score), [precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score), [recall](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score).

To calculate the evaluations metrics is only necessary to run:

The follow results will be shown:

 **Algorithm**  |  **Author** |  **Stage [Run]**                        | **Accuracy** | **F1** | **Precision** | **Recall** | 
----------------|------------|---------------------------------|--------------|--------|---------------|------------|
 KNN    |  Ivanovith Silva | Train [likely-snow-4](https://wandb.ai/ivanovitch-silva/first_image_classifier/runs/2ebl5jzc?workspace=user-ivanovitch-silva) | ?      | ? | ?        | ?     |  
 KNN    | Ivanovitch Silva | Test [confused-sun-5](https://wandb.ai/ivanovitch-silva/first_image_classifier/runs/e8bwl5wq?workspace=user-ivanovitch-silva)  | 0.4533      | 0.4548 | 0.5435        | 0.4533     |
  MLP    |  Morsinaldo Medeiros | Train [likely-snow-4](https://wandb.ai/ivanovitch-silva/first_image_classifier/runs/2ebl5jzc?workspace=user-ivanovitch-silva) | 0.5684      | 0.5662 | 0.5712        | 0.5684     |  
 MLP    | Morsinaldo Medeiros | Test [confused-sun-5](https://wandb.ai/ivanovitch-silva/first_image_classifier/runs/e8bwl5wq?workspace=user-ivanovitch-silva)  | 0.5347      | 0.5343 | 0.5350        | 0.5347     |

 ## Ethical Considerations

 We may be tempted to claim that this dataset contains the only attributes capable of predicting if there is a cat, dog or a panda in an image. However, this is not the case. The dataset is composed of 3,000 images, which is a small number of images to train a model. The dataset is also composed of images with different resolutions, which may lead to a model that is not robust to different image sizes. Also, the dataset does not contain attributes that could infringe someone's rights or harm in any way.

 ## Caveats and Recommendations

 It should be noted that the model trained in this project was used only for validation of a complete data pipeline. It is notary that some important issues related to size of images exist, and adequate techniques need to be adopted in order to balance it. Including data augmentation techniques, for example. Other more appropriate models should be used in order to obtain better results such as Convolutional Neural Networks (CNNs).


