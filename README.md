# Bird Deep Convolutional GAN (Work in Progress)

This is an implementation of a DCGAN which generates images of birds.

[For faster training, it is recommended to run tensorflow on GPU. For reference, it takes several hours to a day (depending on the number of epochs) for this code to run on NVIDIA GeForce GTX 1660. For more information check out https://www.tensorflow.org/install/gpu]

Here are examples of some randomly chosen generated images (64x64 pixels) at epoch 20, 200, and 400:
(This was done in a previous iteration of the code, the current parameters might require a different number of epochs to achieve similar results.)

**Epoch 20:**

![example](generated_images_epoch_20.png)

**Epoch 200:**

![example](generated_images_epoch_200.png)

**Epoch 400:**

![example](generated_images_epoch_400.png)
