# With or Without a mask

## Introduction

Many people have been using machine learning to create very innovative solutions in all areas of science. Some models have up to 99% accuracy, and that's very cool, but in general these models are very heavy, trained for hours or even days on powerful GPUs. The challenge we bring today is: how to run such a model on an embedded device? In general, these devices have limited processing and memory, and should not have high power consumption.

With that in mind, [Mariana Azevedo](https://github.com/marianabritoazevedo) [![Repository](https://img.shields.io/badge/-Repo-191A1B?style=flat-square&logo=github)](https://github.com/marianabritoazevedo/embedded-ai/tree/main/With_Without_Mask), [Thaís Medeiros](https://github.com/thaisaraujo2000) [![Repository](https://img.shields.io/badge/-Repo-191A1B?style=flat-square&logo=github)](https://github.com/thaisaraujo2000/embedded_artificial_intelligence/tree/main/projects/project_2) and I made a small project to embed a model that detects whether or not a person is wearing a mask from a photo into an Arduino Nano 33 BLE Sense provided by [Edge Impulse](https://edgeimpulse.com), which can be seen in the image below.

<p align='center'>
    <img src='./images/kit.png' width=400>
    
Source: [Embarcados (2020)](https://embarcados.com.br/tinyml-machine-learning-para-microcontroladores/)
</p>

You can check out our project using transfer learning done with the GPU in the Google Colaboratory [neste link](../face_mask_detection/)

## The Big Picture

Overall, our project was divided into three steps: The first steps; Train/Test and Deploy. The block diagram below shows the steps of our project:

<p align='center'>
    <img src='./images/big_picture.png' width=500>
</p>


### First Steps

**The Goal**: Our project aims to create a classifier model that, given an image, indicates whether or not a person is wearing a mask.

**Collect, clean and pre-process the data**: The dataset can be found on [Kaggle](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset) and has 12000 images, but only 7298 images were collected. Thus, the data were divided into 80% for training (5838 images) and 20% for testing (1460 images). Some examples can be seen below. The choice of theme is due to the context of the Covid-19 pandemic still being experienced today. 

<p align='center'>
    <img src='./images/data_segregation.png' width=400>
</p>

**Design a model architecture**: The model was built from Edge Impulse's platform and it was chosen to use transfer learning from the MobileNetV1 model through da EON Tuner tool. The architecture of the model can be seen below:

<p align='center'>
    <img src='./images/model_arch.png' width=300>
</p>

## Train/Test

**Train a model**: The training was performed for 20 epochs and the hyperparameter tuning was done using a tool called EON Tuner, which searches for parameters for the model architecture according to some targets defined by the group. The result of the chosen model and the target can be seen in the image below.

<p align='center'>
  <img src="./images/final_model.png" width=300>
</p>

<p align='center'>
  <img src="./images/target.png"  width=300>  
</p>

With this, it is possible to observe that all parameters were within the limitations of the device we want to ship the model and a very good acuity was obtained. It is important to point out that it is good practice that the model occupies about 50% of the total available RAM, since it will be necessary to load the image that will be made the inference during the use of the device.


**Evaluate and Convert**: Once the model is trained, it is time to feed it data that it missed during training. The result of this step can be seen in the step below:

<p align='center'>
    <img src='./images/evaluation.png' width=400>
</p>

Finally, the platform already provides an option to convert the template to the format that the target device will run on.

### Deploy

Once you have downloaded the model, you use the Arduino IDE to load it into the device. As shown in the diagram below, you first add the [.zip](./library_zip/) file of the model as a library and then perform the inference using the BLE 33 camera itself.

The step-by-step on how to deploy can be found in more detail in Marcelo Rovai's course [![Repository](https://img.shields.io/badge/-Repo-191A1B?style=flat-square&logo=github)](https://github.com/Mjrovai/UNIFEI-IESTI01-TinyML-2022.1).

## How to run

It is important to know that these steps on how to run take into account that the entire model creation process has already been carried out on the [Edge Impulse](https://www.edgeimpulse.com/) platform.

1- Clone this repository 

```
git clone https://github.com/marianabritoazevedo/embedded-ai.git
```

2- Download Arduino IDE, you can check it [here](https://www.arduino.cc/en/software)

3- Install the libraries necessary to run this project. They are: `Arduino Mbed OS Nano Boards`, `Harvard_TinyMLx`, `Arduino_TensorFlowLite` and `Arduino_OV767X`.
To install the libraries, you can follow these steps:

```
Tools >> Manage libraries
```

:heavy_exclamation_mark: Attention: to install `Arduino_TensorFlowLite`, it is necessary to use this [repository](https://github.com/tensorflow/tflite-micro-arduino-examples)

4- Take the file `ei-face_mask_detection_2-arduino-1.0.5.zip` in the folder `library_zip` and add as a library in Arduino IDE. This can be done by following the sequence:

```
Sketch >> Include library >> Add .zip library
```

5- Choose the file with the project
```
File >> Examples >> face_mask_detection_2_inferencing >> nano_ble33_sense >> nano_ble33_sense_camera
```

6- Connect your Arduino Nano 33 BLE sense in the Arduino IDE. 
```
Tools >> Board >> Arduino Mbed OS Nano Boads >> Arduino Nano 22 BLE
```

7- Connect the port
```
Tools >> Port >> COM4 (this may change)
```

8- Run the project
```
Verify >> Upload
```

## References

- [Edge Impulse](https://www.edgeimpulse.com)

- [Ivanovitch's repository for embedded artificial intelligence repository](https://github.com/ivanovitchm/embedded.ai)

- [Kaggle | Face mask Dataset](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset)

- [Marcelo Rovai's course](https://github.com/Mjrovai/UNIFEI-IESTI01-TinyML-2022.1)

