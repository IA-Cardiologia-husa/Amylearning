import SimpleITK as sitk
import numpy as np


def read_dicom(path):
	img = sitk.GetArrayFromImage(sitk.ReadImage(path))

	if (len(img.shape)==2):
		img = img[np.newaxis,...,np.newaxis]
	elif (len(img.shape)==3):
		if ((img.shape[-1]==3)or(img.shape[-1]==1)):
			img=img[np.newaxis,...]
		else:
			img=img[...,np.newaxis]
	elif (len(img.shape)!=4):
		print(f"ERROR: Dicom at{sitk.GetArrayFromImage(sitk.ReadImage(path))} has shape {img.shape} while 2,3 or 4 dimensions were expected")

	return img

def write_video(path, array, fps=24):
	writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'PIM1'), fps, (array.shape[1], array.shape[2]))
	for frame in range(array.shape[0]):
		x = _array[frame,...].astype('uint8')
		writer.write(x)
	writer.release()
