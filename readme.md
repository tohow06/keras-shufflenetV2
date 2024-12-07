# Keras ShuffleNet V2

> Keras implementation of ShuffleNet V2

## Example

This example demonstrates the use of ShuffleNet V2 for emotion recognition, based on the [Kaggle Emotion Recognition Challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge). 

A [pre-trained model](./weights/shufflenetv2_emotion_recogn.h5) has been provided, which uses ShuffleNet V2 architecture to predict facial expressions.

Demo result:

![demo](./images/a_emo.png)

## Example Notebooks

The `example` folder contains two useful notebooks to help you get started:

1. **`test.ipynb`**: 
   - Demonstrates how to load and use the pre-trained ShuffleNet V2 model to perform emotion recognition based on the [Kaggle Emotion Recognition Challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge).
   - Walkthrough of loading the model and making predictions on new input images.

2. **`transfer_learning.ipynb`**:
   - Explains how to use transfer learning with the pre-trained ShuffleNet V2 model for a different dataset, specifically the [Food11 Image Dataset](https://www.kaggle.com/datasets/trolukovich/food11-image-dataset/).
   - Includes steps to fine-tune the model and evaluate its performance on the Food11 dataset.

## Environment Compatibility

This implementation has been tested and is compatible with the following environment setup:

- **Keras version:** 2.3.1
- **TensorFlow version:** 2.2.0

### **Important:**  
The code **will not run with the latest versions** of Keras and TensorFlow due to API changes and deprecations. Please ensure you use the tested versions listed above for a smooth experience.

### Using ShuffleNet with the Latest Keras and TensorFlow

If you want to use ShuffleNet V2 with the **latest versions** of Keras and TensorFlow, please switch to the **`main` branch** of this repository.

## References

1. [(Repo) keras-shufflenet](https://github.com/scheckmedia/keras-shufflenet)
2. [(Paper) ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164)

## License

[MIT License](https://opensource.org/licenses/MIT)
