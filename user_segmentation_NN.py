import tensorflow as tf
import numpy as np
import cv2
from .nn_architecture import HyperUNet_1mask
import scipy
import skimage



def resize_img(img_array, new_shape):
	n_frames = img_array.shape[0]
	original_shape = img_array.shape[1:3]

	new_img = np.zeros([*new_shape, n_frames], dtype=float)

	img_array = np.transpose(img_array, [1,2,0])

	if(new_shape[0]/original_shape[0] > new_shape[1]/original_shape[1]):
		resize_shape = [new_shape[0], new_shape[1]]
		shift = (original_shape[1]-int(new_shape[1]*original_shape[0]/new_shape[0]))
		if (shift !=0):
			sliced_img = img_array[:,shift//2:(shift//2-shift)]
		new_img[0:resize_shape[0], 0:resize_shape[1],:] = cv2.resize(sliced_img, dsize = (resize_shape[1], resize_shape[0]), interpolation=cv2.INTER_LINEAR)
	else:
		resize_shape = [new_shape[0], int(original_shape[1]*new_shape[0]/original_shape[0])]
		new_img[0:resize_shape[0], 0:resize_shape[1],:] = cv2.resize(img_array, dsize = (resize_shape[1], resize_shape[0]), interpolation=cv2.INTER_LINEAR)
		shift = (new_shape[1]-int(original_shape[1]*new_shape[0]/original_shape[0]))//2
		new_img = np.roll(new_img, shift, axis = 1)

	new_img = np.transpose(new_img, [2,0,1])

	return new_img

def resize_mask(mask_array, new_shape):
	n_frames = mask_array.shape[0]
	original_shape = mask_array.shape[1:3]

	new_mask = np.zeros([*new_shape, n_frames], dtype=int)


	mask_array = np.transpose(mask_array, [1,2,0])

	if(new_shape[0]/original_shape[0] > new_shape[1]/original_shape[1]):
		resize_shape = [new_shape[0], new_shape[1]]
		shift = (original_shape[1]-int(new_shape[1]*original_shape[0]/new_shape[0]))
		if (shift !=0):
			sliced_mask = mask_array[:,shift//2:(shift//2-shift)]
		new_mask[0:resize_shape[0], 0:resize_shape[1],:] = cv2.resize(sliced_mask, dsize = (resize_shape[1], resize_shape[0]), interpolation=cv2.INTER_NEAREST)
	else:
		resize_shape = [new_shape[0], int(original_shape[1]*new_shape[0]/original_shape[0])]
		new_mask[0:resize_shape[0], 0:resize_shape[1],:] = cv2.resize(mask_array, dsize = (resize_shape[1], resize_shape[0]), interpolation=cv2.INTER_NEAREST)
		shift = (new_shape[1]-int(original_shape[1]*new_shape[0]/original_shape[0]))//2
		new_mask = np.roll(new_mask, shift, axis = 1)

	new_mask = np.transpose(new_mask, [2,0,1])

	return new_mask

def postprocess_1mask(pred, dilation_size = 21, erosion_size = 21):
	#dilatamos, rellenamos huecos y erosionamos para quitar pequeños huecos y 'golfos'
	for frame in range(pred.shape[0]):
		pred[frame,...] = skimage.morphology.binary_dilation(pred[frame,...], np.ones([dilation_size,dilation_size]))
		pred[frame,...] = scipy.ndimage.binary_fill_holes(pred[frame,...])
		pred[frame,...] = skimage.morphology.binary_erosion(pred[frame,...], np.ones([dilation_size,dilation_size]))

		#erosionamos y luego dilatamos para quitar pequeñas islas y 'cabos'
		pred[frame,...] = skimage.morphology.binary_erosion(pred[frame,...], np.ones([erosion_size,erosion_size]))
		pred[frame,...] = skimage.morphology.binary_dilation(pred[frame,...], np.ones([erosion_size,erosion_size]))

		#Nos quedamos con la componente conectada más grande
		labels = skimage.measure.label(pred[frame,...], background=0, return_num=False, connectivity=1)
		if(labels.max() !=0):
			max_label = np.argmax(np.bincount(labels.flat)[1:])+1
			pred[frame,...] = (labels == max_label)
		else:
			pred[frame,...] = np.zeros(labels.shape)

	return pred

# class UserSegmenter:
# 	def __init__(self):
# 		self.neural_network = HyperUNet2Masks(input_size = (384, 384,1),
# 							depth = 5,
# 							stack_size = 2,
# 							filter_size = 5,
# 							filters_base = 10,
# 							filter_grow = 2,
# 							depth_batch_norm = True,
# 							stack_batch_norm = False,
# 							stack_activation = None,
# 							depth_activation = 'relu',
# 							upsampling_type = 'upsampling',
# 							kernel_regularizer = None,
# 							bias_regularizer =  None
# 							)
#
# 		self.neural_network.load_weights('model2maks9413.h5')
#
# 	def segment(self, img):
# 		img = img[...,0]
# 		img = (img-img.min())/(img.max()-img.min())
#
# 		original_shape = img.shape[1:3]
#
# 		img_resized = resize_img(img, (384,384))
#
# 		pred = (self.neural_network.predict(img_resized)>0.5).astype(int)
# 		pred = pred[...,0]*3+pred[...,1]*2 - pred[...,0]*pred[...,1]*4
#
# 		pred_resized = resize_mask(pred, original_shape)
#
# 		return pred_resized

class UserSegmenter_old:
	def __init__(self):
		self.neural_network = HyperUNet_1mask(input_size = (384, 384,1),
							   depth = 5,
							   stack_size = 2,
							   filter_size = 5,
							   filters_base = 10,
							   filter_grow = 2,
							   depth_batch_norm = True,
							   stack_batch_norm = False,
							   stack_activation = None,
							   depth_activation = 'relu',
							   upsampling_type = 'upsampling',
							   kernel_regularizer = None,
							   bias_regularizer =  None
							   )

		self.neural_network.load_weights('UNet1MaskTrainOnEverything2.h5')

	def segment(self, img, threshold = 0.5):
		img = img[...,0]
		img = (img-img.min())/(img.max()-img.min())

		original_shape = img.shape[1:3]

		img_resized = resize_img(img, (384,384))

		pred = (self.neural_network.predict(img_resized, batch_size = 10)>threshold)[...,0]

		pred = self.preprocess_1mask(pred).astype(int)
		pred_resized = resize_mask(pred, original_shape)

		return pred_resized

	def preprocess_1mask(self, pred, dilation_size = 21, erosion_size = 21):
		#dilatamos, rellenamos huecos y erosionamos para quitar pequeños huecos y 'golfos'
		for frame in range(pred.shape[0]):
			pred[frame,...] = skimage.morphology.binary_dilation(pred[frame,...], np.ones([dilation_size,dilation_size]))
			pred[frame,...] = scipy.ndimage.binary_fill_holes(pred[frame,...])
			pred[frame,...] = skimage.morphology.binary_erosion(pred[frame,...], np.ones([dilation_size,dilation_size]))

			#erosionamos y luego dilatamos para quitar pequeñas islas y 'cabos'
			pred[frame,...] = skimage.morphology.binary_erosion(pred[frame,...], np.ones([erosion_size,erosion_size]))
			pred[frame,...] = skimage.morphology.binary_dilation(pred[frame,...], np.ones([erosion_size,erosion_size]))

			#Nos quedamos con la componente conectada más grande
			labels = skimage.measure.label(pred[frame,...], background=0, return_num=False, connectivity=1)
			if(labels.max() !=0):
				max_label = np.argmax(np.bincount(labels.flat)[1:])+1
				pred[frame,...] = (labels == max_label)
			else:
				pred[frame,...] = np.zeros(labels.shape)

		return pred

class UserSegmenter:
	def __init__(self):
		self.neural_network = HyperUNet_1mask(input_size = (384, 384,1),
							   depth = 5,
							   stack_size = 2,
							   filter_size = 3,
							   filters_base = 10,
							   filter_grow = 2,
							   depth_batch_norm = True,
							   stack_batch_norm = False,
							   stack_activation = 'relu',
							   depth_activation = 'relu',
							   upsampling_type = 'upsampling',
							   kernel_regularizer = None,
							   bias_regularizer =  None
							   )

		self.neural_network.load_weights('UNet_2022-12-16_205126.h5')

	def segment(self, img, threshold = 0.5):
		img = img[...,0]
		img = (img-img.min())/(img.max()-img.min())

		original_shape = img.shape[1:3]

		img_resized = resize_img(img, (384,384))

		pred = (self.neural_network.predict(img_resized, batch_size = 10)>threshold)[...,0]

		pred = self.preprocess_1mask(pred).astype(int)
		pred_resized = resize_mask(pred, original_shape)

		return pred_resized

	def preprocess_1mask(self, pred, dilation_size = 21, erosion_size = 21):
		#dilatamos, rellenamos huecos y erosionamos para quitar pequeños huecos y 'golfos'
		for frame in range(pred.shape[0]):
			pred[frame,...] = skimage.morphology.binary_dilation(pred[frame,...], np.ones([dilation_size,dilation_size]))
			pred[frame,...] = scipy.ndimage.binary_fill_holes(pred[frame,...])
			pred[frame,...] = skimage.morphology.binary_erosion(pred[frame,...], np.ones([dilation_size,dilation_size]))

			#erosionamos y luego dilatamos para quitar pequeñas islas y 'cabos'
			pred[frame,...] = skimage.morphology.binary_erosion(pred[frame,...], np.ones([erosion_size,erosion_size]))
			pred[frame,...] = skimage.morphology.binary_dilation(pred[frame,...], np.ones([erosion_size,erosion_size]))

			#Nos quedamos con la componente conectada más grande
			labels = skimage.measure.label(pred[frame,...], background=0, return_num=False, connectivity=1)
			if(labels.max() !=0):
				max_label = np.argmax(np.bincount(labels.flat)[1:])+1
				pred[frame,...] = (labels == max_label)
			else:
				pred[frame,...] = np.zeros(labels.shape)

		return pred
