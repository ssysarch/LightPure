# LightPure: Realtime Adversarial Image Purification for Mobile Devices Using Diffusion Models

Abstract: _Autonomous mobile systems increasingly rely on deep neural networks for perception and decision-making. While effective, these systems are vulnerable to adversarial machine learning attacks. Common countermeasures include leveraging adversarial training and/or data or network transformation. Although widely adopted, the major shortcoming in these countermeasures is that these methods need complete and invasive access to the classifiers, which are usually proprietary. Furthermore, the cost of (re-) training is often too expensive for large models.
We introduces an innovative approach that significantly enhances the purification of adversarial images, achieving parity with the state-of-the-art purification method in terms of accuracy, while offering substantial improvements in speed and computational efficiency which is suitable for resource-constrained mobile devices. Our method leverages a Generative Adversarial Network (GAN) framework for purification, optimized for rapid processing without sacrificing the model's lightweight nature. We propose several contributions in designing our model to achieve a reasonable balance between classification accuracy and adversarial robustness while maintaining a desired latency. We design and implement a proof-of-concept and evaluate our method using several attack scenarios and datasets. Our results show that LightPure can outperform existing purification methods by 2-10x in terms of latency while achieving higher accuracy and robustness for various black-, gray-, and white-box attack scenarios._

# Dataset preparation and pre-trained models

We trained on two datasets CIFAR-10 [Download](https://www.cs.toronto.edu/~kriz/cifar.html) and GTSRB [Download](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign).
For our classifier, we used pretrained Resnet-56 [Download](https://github.com/chenyaofo/pytorch-cifar-models?tab=readme-ov-file) for CIFAR-10 and self-trained Resnet-18 for GTSRB.
We resized GTSRB to 32x32 for training and evaluation.

# Running code 

Python version 3.10.13
PyTorch version 2.1.0 with CUDA version 12.1

To run the train, hyperparameters may be specified.
```python train.py --ch_mult 1 2 2 4 --beta_max .2 --beta_min .05 --saved_model ....```

