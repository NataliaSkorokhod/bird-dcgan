import os
import glob
import tensorflow as tf
from matplotlib import pyplot
import numpy as np
from numpy.random import randn
from numpy.random import randint
import time
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, LeakyReLU, ZeroPadding2D, BatchNormalization,\
	Reshape, Activation, UpSampling2D
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import cv2


"""
The dataset used for this project is adapted from:
https://www.kaggle.com/gpiosenka/100-bird-species


Generator architecture:

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_2 (Dense)              (None, 32768)             4947968   
_________________________________________________________________
reshape_1 (Reshape)          (None, 16, 16, 128)       0         
_________________________________________________________________
up_sampling2d_1 (UpSampling2 (None, 32, 32, 128)       0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 32, 32, 64)        73792     
_________________________________________________________________
batch_normalization_4 (Batch (None, 32, 32, 64)        256       
_________________________________________________________________
leaky_re_lu_4 (LeakyReLU)    (None, 32, 32, 64)        0         
_________________________________________________________________
up_sampling2d_2 (UpSampling2 (None, 64, 64, 64)        0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 64, 64, 32)        18464     
_________________________________________________________________
batch_normalization_5 (Batch (None, 64, 64, 32)        128       
_________________________________________________________________
leaky_re_lu_5 (LeakyReLU)    (None, 64, 64, 32)        0         
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 64, 64, 16)        4624      
_________________________________________________________________
batch_normalization_6 (Batch (None, 64, 64, 16)        64        
_________________________________________________________________
leaky_re_lu_6 (LeakyReLU)    (None, 64, 64, 16)        0         
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 64, 64, 3)         435       
_________________________________________________________________
activation_1 (Activation)    (None, 64, 64, 3)         0         
=================================================================


Discriminator architecture:

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 32, 32, 64)        1792      
_________________________________________________________________
zero_padding2d_1 (ZeroPaddin (None, 33, 33, 64)        0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 33, 33, 64)        256       
_________________________________________________________________
leaky_re_lu_1 (LeakyReLU)    (None, 33, 33, 64)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 33, 33, 64)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 17, 17, 128)       73856     
_________________________________________________________________
batch_normalization_2 (Batch (None, 17, 17, 128)       512       
_________________________________________________________________
leaky_re_lu_2 (LeakyReLU)    (None, 17, 17, 128)       0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 17, 17, 128)       0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 17, 17, 256)       295168    
_________________________________________________________________
batch_normalization_3 (Batch (None, 17, 17, 256)       1024      
_________________________________________________________________
leaky_re_lu_3 (LeakyReLU)    (None, 17, 17, 256)       0         
_________________________________________________________________
dropout_3 (Dropout)          (None, 17, 17, 256)       0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 73984)             0         
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 73985     
=================================================================
Total params: 446,593
Trainable params: 445,697
Non-trainable params: 896

"""


tf.config.experimental.set_memory_growth = True				# This prevents a memory access error if the GPU is
															# being used by other processes
tf.compat.v1.GPUOptions.per_process_gpu_memory_fraction = 0.9

# Load the dataset
def load_data(path, size):
	data_list = list()
	for filename in glob.iglob(path, recursive=True):
		if os.path.isfile(filename):
			pixels = load_img(filename, target_size=size)
			pixels = img_to_array(pixels)
			pixels = pixels*(2./255) - 1		# Normalizing to [-1, 1]
			data_list.append(pixels)
	return np.asarray(data_list)

# Choose n real samples from the data
def choose_real_samples(data, n):
	indices = randint(0, data.shape[0], n)		# Choose n random indices
	X = data[indices]
	Y = np.ones((n, 1))							# Generate "real" class labels
	return X, Y

# Generate random vectors to be fed into the generator
# n - number of samples, dim - dimension of the vector space
def generate_random_vectors(dim , n):
	input = randn(dim * n)						# Using a normal distribution as recommended
	input = input.reshape(n, dim)
	return input

# Use the generator to create n samples from random vectors
def create_fake_samples(generator, dim, n):
	input = generate_random_vectors(dim, n)
	X = generator.predict(input)
	Y = np.zeros((n, 1))						# Generate "fake" class labels
	return X, Y

def create_discriminator():
	model = Sequential()
	model.add(Conv2D(64, kernel_size=3, strides=2, padding="same", input_shape=(64, 64, 3)))
	model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
	model.add(BatchNormalization(momentum=0.8))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.25))
	model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
	model.add(BatchNormalization(momentum=0.8))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.25))
	model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
	model.add(BatchNormalization(momentum=0.8))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))
	optimizer = Adam(lr=0.0004, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	model.summary()
	return model

def create_generator(dim):
	model = Sequential()
	model.add(Dense(128 * 16 * 16, activation="relu", input_dim=dim))
	model.add(Reshape((16, 16, 128)))
	model.add(UpSampling2D())
	model.add(Conv2D(64, kernel_size=3, padding="same"))
	model.add(BatchNormalization(momentum=0.8))
	model.add(LeakyReLU(alpha=0.2))					# The generator can also use ReLU, as vanishing gradients are less common here
	model.add(UpSampling2D())
	model.add(Conv2D(32, kernel_size=3, padding="same"))
	model.add(BatchNormalization(momentum=0.8))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2D(16, kernel_size=3, padding="same"))
	model.add(BatchNormalization(momentum=0.8))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2D(3, kernel_size=3, padding="same"))
	model.add(Activation("tanh"))
	model.summary()
	return model

# Combine the generator and discriminator into a single model (intended for generator update)
def GAN(generator, discriminator):
	discriminator.trainable = False					# Setting the discriminator to not update weights
	model = Sequential()
	model.add(generator)
	model.add(discriminator)
	optimizer = Adam(lr=0.0001, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=optimizer)
	model.summary()
	return model

def evaluate_GAN(epoch, generator, discriminator, data, dim, samples=100):
	X_real, Y_real = choose_real_samples(data, samples)
	X_fake, Y_fake = create_fake_samples(generator, samples, dim)
	_, accuracy_real = discriminator.evaluate(X_real, Y_real, verbose=0)
	_, accuracy_fake = discriminator.evaluate(X_fake, Y_fake, verbose=0)
	print('accuracy real: %d, accuracy fake: %d' % (accuracy_real, accuracy_fake))

	# Create an n*n grid of generated images and saving it
	n = 3
	for i in range(n**2):
		pyplot.subplot(n, n, i + 1)
		pyplot.axis('off')
		image = cv2.normalize(X_fake[i], None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
		pyplot.imshow(image)
	filename = 'generated_images_epoch_%d.png' % (epoch + 1)
	pyplot.savefig(filename, bbox_inches='tight')

	# Save the current version of the generator
	filename = 'generator_model_%03d.h5' % (epoch + 1)
	generator.save(filename)


def plot(epoch, disc_real, disc_fake, gen, acc_real, acc_fake):
	# Plot loss (Last 100 values)
	pyplot.figure(figsize=(20, 20))
	pyplot.subplot(2, 1, 1)
	pyplot.plot(disc_real[-100:], label='disc_real')
	pyplot.plot(disc_fake[-100:], label='disc_fake')
	pyplot.plot(gen[-100:], label='gen')
	pyplot.title('Discriminator and Generator Loss')
	pyplot.legend()
	# Plot discriminator accuracy (Last 100 values)
	pyplot.subplot(2, 1, 2)
	pyplot.plot(acc_real[-100:], label='acc-real')
	pyplot.plot(acc_fake[-100:], label='acc-fake')
	pyplot.title('Discriminator Accuracy')
	pyplot.legend()
	# Save plot
	pyplot.savefig('loss_and_accuracy_at_epoch_%d.png' %(epoch + 1), bbox_inches='tight')
	pyplot.close()


# Train the generator and discriminator
def train(generator, discriminator, GAN, data, dim, epochs, batch_size):
	batches_per_epoch = int(data.shape[0] / batch_size)
	half_batch = int(batch_size/2)

	# Recording loss and accuracy:
	# Discriminator real loss, discriminator fake loss, generator loss, discriminator real accuracy,
	# discriminator fake accuracy.
	disc_real, disc_fake, gen, acc_real, acc_fake = list(), list(), list(), list(), list()

	for i in range(epochs):
		for j in range(batches_per_epoch):
			X_real, Y_real = choose_real_samples(data, half_batch)
			discriminator_loss_real, discriminator_accuracy_real = discriminator.train_on_batch(X_real, Y_real)
			X_fake, Y_fake = create_fake_samples(generator, dim, half_batch)
			discriminator_loss_fake, discriminator_accuracy_fake = discriminator.train_on_batch(X_fake, Y_fake)
			X_GAN = generate_random_vectors(dim, batch_size)
			Y_GAN = np.ones((batch_size, 1))
			generator_loss = GAN.train_on_batch(X_GAN, Y_GAN)

			disc_real.append(discriminator_loss_real)
			disc_fake.append(discriminator_loss_fake)
			gen.append(generator_loss)
			acc_real.append(discriminator_accuracy_real)
			acc_fake.append(discriminator_accuracy_fake)

			print('>%d, %d/%d, disc_loss_real=%.3f, disc_loss_fake=%.3f gen_loss=%.3f, disc_accuracy_real=%.3f, '
				  'disc_accuracy_fake=%.3f' %
				(i+1, j+1, batches_per_epoch, discriminator_loss_real, discriminator_loss_fake, generator_loss,
				 discriminator_accuracy_real, discriminator_accuracy_fake))

		if (i+1)%20 == 0:
			evaluate_GAN(i, generator, discriminator, data, dim, samples=100)
			plot(i, disc_real, disc_fake, gen, acc_real, acc_fake)
			disc_real, disc_fake, gen, acc_real, acc_fake = list(), list(), list(), list(), list()


if __name__ == '__main__':

	dim = 150									# The dimension of the vector space from which we take random vectors

	load_new_data = False						# Change to True if loading new data is required

	new_data_path = None						# Give as '[path]\**' for loading files from all subfolders. For example:
												# 'C:\\Users\\Natalia\PycharmProjects\GAN\\consolidated\\Small Birds\\**'

	new_data_name = None						# Under which name to save the data, e.g 'birds_data_64.npy'

	data_file = 'birds_data_64.npy'				# Existing numpy file from which the data is loaded (alternative to the above)
	image_size = (64, 64)						# Images will be scaled to the given size

	discriminator = create_discriminator()
	generator = create_generator(dim)
	GAN = GAN(generator, discriminator)

	if load_new_data:
		print('loading data')
		data = load_data(new_data_path, size=image_size)
		print('loaded data')
		np.save(new_data_name, data)

	else:
		data = np.load(data_file)


	time0 = time.time()
	train(generator, discriminator, GAN, data, dim=dim, epochs=400, batch_size=128)  # Batch size should be even (preferably a power of 2, for efficiency)
	time1 = time.time()
	print('The training took %.3f minutes') % (time1-time0)/60
