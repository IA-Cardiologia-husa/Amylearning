import numpy as np
import cv2
import skimage
import scipy

def random_augmentation(img,
						seed,
						rotation_angle = 20,
						shear_angle = 5,
						gamma_range = [0.66,1.5],
						zoom_range = [0.9,1.1],
						reflection = True):
	np.random.seed(seed)
	rotation = 2*np.random.random()*rotation_angle-rotation_angle
	shear = 2*np.random.random()*shear_angle-shear_angle
	zoom = np.random.random()*(zoom_range[1]-zoom_range[0])+zoom_range[0]
	gamma = np.random.random()*(gamma_range[1]-gamma_range[0])+gamma_range[0]
	reflection = (np.random.random()>0.5) and reflection

	print('forward transform', seed, rotation, shear, zoom, gamma, reflection)

	return transform_image(img, rotation, shear, zoom, gamma, reflection)

def restore_augmentation(mask,
						seed,
						rotation_angle = 20,
						shear_angle = 5,
						gamma_range = [0.66,1.5],
						zoom_range = [0.9,1.1],
						reflection = True):
	np.random.seed(seed)
	rotation = 2*np.random.random()*rotation_angle-rotation_angle
	shear = 2*np.random.random()*shear_angle-shear_angle
	zoom = np.random.random()*(zoom_range[1]-zoom_range[0])+zoom_range[0]
	gamma = np.random.random()*(gamma_range[1]-gamma_range[0])+gamma_range[0]
	reflection = ((np.random.random()>0.5) and reflection)

	print('inverse transform', seed, rotation, shear, zoom, gamma, reflection)

	return inverse_transform_mask(mask, rotation, shear, zoom, reflection)



def transform_image(img, rotation, shear, zoom, gamma, reflection):
	img = np.transpose(img[...,0], [1,2,0])
	img = img ** (1/gamma)

	M1 = np.identity(3)
	M1[:2,:] = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2),rotation,1)

	M2 = np.identity(3)
	h = img.shape[0]
	w = img.shape[1]
	pts1 = np.float32([[0,w],[h,w],[0,0]])
	pts2 = np.float32([[0,w],[h,w],[h*np.tan(shear/180*np.pi),0]])
	M2[:2,:] = cv2.getAffineTransform(pts1,pts2)

	M3 = np.identity(3)
	z= (zoom-1)/2
	pts1 = np.float32([[0,1],[1,1],[0,0]])
	pts2 = np.float32([[-z,1+z],[1+z,1+z],[-z,-z]])
	M3[:2,:] = cv2.getAffineTransform(pts1,pts2)

	M = M3 @ M1 @ M2

	img = cv2.warpAffine(img,M[:2,:],(img.shape[1],img.shape[0]),flags=cv2.INTER_LINEAR)

	if(reflection):
		img = np.flip(img, axis=1)

	img = np.transpose(img, [2,0,1])
	return img[..., np.newaxis]

def inverse_transform_mask(mask, rotation, shear, zoom, reflection):
	mask = np.transpose(mask, [1,2,0])

	if(reflection):
		mask = np.flip(mask, axis=1)

	M1 = np.identity(3)
	M1[:2,:] = cv2.getRotationMatrix2D((mask.shape[1]/2, mask.shape[0]/2),-rotation,1)

	M2 = np.identity(3)
	h = mask.shape[0]
	w = mask.shape[1]
	pts1 = np.float32([[0,w],[h,w],[0,0]])
	pts2 = np.float32([[0,w],[h,w],[h*np.tan(-shear/180*np.pi),0]])
	M2[:2,:] = cv2.getAffineTransform(pts1,pts2)

	M3 = np.identity(3)
	z= (1/zoom-1)/2
	pts1 = np.float32([[0,1],[1,1],[0,0]])
	pts2 = np.float32([[-z,1+z],[1+z,1+z],[-z,-z]])
	M3[:2,:] = cv2.getAffineTransform(pts1,pts2)

	M = M2 @ M1 @ M3

	mask = cv2.warpAffine(mask,M[:2,:],(mask.shape[1],mask.shape[0]),flags=cv2.INTER_NEAREST)

	mask = np.transpose(mask, [2,0,1])

	return mask

#TODO: postprocess_prediction for images with only lv?
def postprocess_prediction(pred, dilation_erosion_la_lv=51):
	for frame in range(pred.shape[0]):
		la_lv = (pred[frame,...,0]==3) | (pred[frame,...,0]==1)
		myo_lv = (pred[frame,...,0]==2) | (pred[frame,...,0]==1)

		labels = skimage.measure.label(myo_lv, background=0, return_num=False, connectivity=1)
		if(labels.max() !=0):
			max_label = np.argmax(np.bincount(labels.flat)[1:])+1
			myo_lv = (labels == max_label)
		else:
			myo_lv = np.zeros(labels.shape)
		myo_lv = skimage.morphology.convex_hull_image(myo_lv)

		labels = skimage.measure.label(la_lv, background=0, return_num=False, connectivity=1)
		if(labels.max() !=0):
			max_label = np.argmax(np.bincount(labels.flat)[1:])+1
			la_lv = (labels == max_label)
		else:
			la_lv = np.zeros(labels.shape)
		la_lv = skimage.morphology.binary_dilation(la_lv, np.ones([dilation_erosion_la_lv,dilation_erosion_la_lv]))
		la_lv = scipy.ndimage.binary_fill_holes(la_lv)
		la_lv = skimage.morphology.binary_erosion(la_lv, np.ones([dilation_erosion_la_lv,dilation_erosion_la_lv]))

		pred[frame,...,0] = la_lv.astype(int)*3+myo_lv.astype(int)*2-(la_lv & myo_lv).astype(int)*4


		#we take the convex hull of each predicted region that does not intersect other regions

		lv = (pred[frame,...,0]==1)
		myo = (pred[frame,...,0]==2)
		la = (pred[frame,...,0]==3)

		lv_ch = np.logical_and(skimage.morphology.convex_hull_image(lv),
							   ~np.logical_or(myo, la))
		la_lv_ch = np.logical_and(skimage.morphology.convex_hull_image(np.logical_or(la, lv_ch)),
							   ~myo)
	#     la_ch = np.logical_and(~lv_ch, la_lv_ch)

		myo_ch = np.logical_and(skimage.morphology.convex_hull_image(myo),
								~la_lv_ch)


		#lv largest connected component
		labels = skimage.measure.label(lv_ch, background=0, return_num=False, connectivity=1)
		if(labels.max() !=0):
			max_label = np.argmax(np.bincount(labels.flat)[1:])+1
			lv_mask = (labels == max_label)
		else:
			lv_mask = np.zeros(labels.shape)
		lv_mask = scipy.ndimage.binary_fill_holes(lv_mask)

		#myo largest connected component next to lv
		labels = skimage.measure.label(np.logical_or(myo_ch, lv_mask), background=0, return_num=False, connectivity=1)
		if(labels.max() !=0):
			max_label = np.argmax(np.bincount(labels.flat)[1:])+1
			myo_mask = (labels == max_label)
		else:
			myo_mask = np.zeros(labels.shape)
		myo_mask = scipy.ndimage.binary_fill_holes(myo_mask)
	#     myo_mask = np.logical_xor(myo_mask, lv_mask)

		lv_myo_mask = scipy.ndimage.binary_fill_holes(np.logical_or(lv_mask, myo_mask))

		la_ch = skimage.morphology.convex_hull_image(np.logical_or(la, np.logical_and(skimage.morphology.dilation(lv_mask), ~lv_myo_mask)))
		la_ch = np.logical_and(la_ch, ~lv_myo_mask)


		#la largest connected component
		labels = skimage.measure.label(la_ch, background=0, return_num=False, connectivity=1)
		if(labels.max() !=0):
			max_label = np.argmax(np.bincount(labels.flat)[1:])+1
			la_mask = (labels == max_label)
		else:
			la_mask = np.zeros(labels.shape)
		la_mask = scipy.ndimage.binary_fill_holes(la_mask)
		la_mask = np.logical_and(la_mask, ~lv_myo_mask)

		lv_la_mask = scipy.ndimage.binary_fill_holes(np.logical_or(lv_mask, la_mask))

#     processed_pred = np.logical_or(lv_myo_mask, lv_la_mask.astype(int))+myo_mask.astype(int)+2*la_mask.astype(int)
		pred[frame,...,0] = 2 * lv_myo_mask.astype(int) + 3*lv_la_mask.astype(int)-4*np.logical_and(lv_myo_mask, lv_la_mask).astype(int)

	return pred
