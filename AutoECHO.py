import os
import numpy as np
import pandas as pd
import datetime as dt
import pydicom
import pickle
import logging
import sys
import h5py
import luigi
import cv2
import matplotlib.pyplot as plt
import scipy
import io

from skimage import measure, morphology,draw

from .utils.segmentutils import *
from .utils.fileutils import *
from .utils.trackingutils import *
from .utils.geometryutils import *
from .user_segmentation_NN import *

# Global variables for path folders
log_path = os.path.abspath("log")
tmp_folder = os.path.abspath("tmp")
dicom_folder = os.path.abspath("dcms")
report_folder = os.path.abspath("reports")

def setupLog(name):
	try:
		os.makedirs(log_path)
	except:
		pass
	logging.basicConfig(
		level=logging.DEBUG,
		format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
		filename=os.path.join(log_path, f'{name}.log'),
		filemode='a'
		)

	stdout_logger = logging.getLogger(f'STDOUT_{name}')
	sl = StreamToLogger(stdout_logger, logging.INFO)
	sys.stdout = sl

	stderr_logger = logging.getLogger(f'STDERR_{name}')
	sl = StreamToLogger(stderr_logger, logging.ERROR)
	sys.stderr = sl

class StreamToLogger(object):
	"""
	Fake file-like stream object that redirects writes to a logger instance.
	"""
	def __init__(self, logger, log_level=logging.INFO):
		self.logger = logger
		self.log_level = log_level
		self.linebuf = ''

	def write(self, buf):
		for line in buf.rstrip().splitlines():
			self.logger.log(self.log_level, line.rstrip())

	def flush(self):
		pass

class InfoDCM(luigi.Task):
	dcm_name=luigi.Parameter()

	def run(self):
		file_path = os.path.join(dicom_folder,self.dcm_name+'.dcm')

		dataset = pydicom.dcmread(file_path)

		info = {}
		info["fps"] = None
		info["frames"] = None
		info["rows"] = None
		info["columns"] = None
		info["min x0"] = None
		info["min y0"] = None
		info["max x1"] = None
		info["max y1"] = None
		info["pixel spacing x"] = None
		info["pixel spacing y"] = None
		info["heart rate"] = None

		available_tags = list(dataset.keys())

		if ('0018','0040') in available_tags:
			info["fps"] = dataset[('0018','0040')].value
		elif ('0018','1063') in available_tags:
			info["fps"] = 1000 / dataset[('0018','1063')].value
		elif ('0018','1065') in available_tags:
			info["fps"] = 1000/np.array(dataset[('0018','1065')].value[1:-1]).mean()

		if ('0028','0008') in available_tags:
			info["frames"] = dataset[('0028','0008')].value

		if ('0028','0010') in available_tags:
			info["rows"] = dataset[('0028','0010')].value

		if ('0028','0011') in available_tags:
			info["columns"] = dataset[('0028','0011')].value

		if ('0018','6011') in available_tags:
			seq = dataset[('0018','6011')]
			for item in seq:
				available_seq_tags = list(item.keys())
				if ('0018','6014') in available_seq_tags:
					if item[('0018','6014')].value == 1:
						try:
							info["min x0"] = item[('0018','6018')].value
							info["min y0"] = item[('0018','601A')].value
							info["max x1"] = item[('0018','601C')].value
							info["max y1"] = item[('0018','601E')].value
						except:
							pass

						try:
							info['pixel spacing x'] = item[('0018','602C')].value
							info['pixel spacing y'] = item[('0018','602E')].value
						except:
							pass

		if (info['pixel spacing x'] == None) and (('0028','0030') in available_tags):
			info['pixel spacing x'] = dataset[('0028','0030')].value[0]

		if (info['pixel spacing y'] == None) and (('0028','0030') in available_tags):
			info['pixel spacing y'] = dataset[('0028','0030')].value[1]

		if ('0018','1088') in available_tags:
			info["heart rate"] = dataset[('0018','1088')].value


		df = pd.DataFrame([info])
		df.to_csv(self.output()['csv'].path, index=False)
		with open(self.output()['pickle'].path, 'wb') as handle:
			pickle.dump(info, handle, protocol=pickle.HIGHEST_PROTOCOL)

	def output(self):
		try:
			os.makedirs(os.path.join(report_folder, self.dcm_name,self.__class__.__name__))
		except:
			pass

		return {'csv': luigi.LocalTarget(os.path.join(report_folder, self.dcm_name,self.__class__.__name__, f"info.csv")),
				'pickle': luigi.LocalTarget(os.path.join(report_folder, self.dcm_name,self.__class__.__name__, f"info.pickle"))}

class MeasurementsVolumesDCM(luigi.Task):
	dcm_name=luigi.Parameter()

	def requires(self):
		return {'volumes':VolumeCurvesDCM(dcm_name = self.dcm_name),
				'info':InfoDCM(dcm_name = self.dcm_name)}

	def run(self):
		setupLog(self.__class__.__name__)

		try:
			os.makedirs(os.path.join(tmp_folder,self.__class__.__name__))
		except:
			pass

		g = h5py.File(self.input()['volumes']['h5'].path, 'r')
		with open(self.input()['volumes']['dyas_sys_list'].path, 'r') as handle:
			lines = handle.read().splitlines()
			try:
				dyas_list = lines[0].split(',')
			except:
				dyas_list = []
			try:
				sys_list = lines[1].split(',')
			except:
				sys_list = []
			dyas_list = np.array(dyas_list, dtype=float).astype(int)
			sys_list = np.array(sys_list, dtype=float).astype(int)

		with open(self.input()['info']['pickle'].path, 'rb') as handle:
			info = pickle.load(handle)

		if info["fps"] is None:
			fps = 50
		else:
			fps = info["fps"]

		vol_curve = g["volume curve smoothed"][...]


		measurements = {}
		measurements['LVEF (automated)'] = (vol_curve.max()-vol_curve.min())/(vol_curve.max()+1e-7)
		measurements['Dyastolic volume (automated)'] = vol_curve.max()
		measurements['Systolic volume (automated)'] = vol_curve.min()

		volume_diff = vol_curve[1:]-vol_curve[:-1]
		measurements['peak_vol_relative_contraction_rate'] = volume_diff.min()*fps/(vol_curve.max()+1e-7)
		measurements['peak_vol_relative_distention_rate'] = volume_diff.max()*fps/(vol_curve.max()+1e-7)
		measurements['peak_vol_absolute_contraction_rate'] = volume_diff.min()*fps
		measurements['peak_vol_absolute_distention_rate'] = volume_diff.max()*fps

		df = pd.DataFrame([measurements])
		df.to_csv(self.output().path, index=False)

	def output(self):
		try:
			os.makedirs(os.path.join(report_folder, self.dcm_name,self.__class__.__name__))
		except:
			pass

		return luigi.LocalTarget(os.path.join(report_folder, self.dcm_name,self.__class__.__name__, f"measurements_volume.csv"))

class MeasurementsGT(luigi.Task):
	dcm_name=luigi.Parameter()

	def requires(self):
		return {'gt':GTMasksAndVolumes(dcm_name = self.dcm_name),
				'info':InfoDCM(dcm_name = self.dcm_name)}

	def run(self):
		setupLog(self.__class__.__name__)

		try:
			os.makedirs(os.path.join(tmp_folder,self.__class__.__name__))
		except:
			pass

		g = h5py.File(self.input()['gt']['h5'].path, 'r')


		with open(self.input()['info']['pickle'].path, 'rb') as handle:
			info = pickle.load(handle)

		measurements = {}

		if "dyastole volume ground truth" in g.keys():
			measurements['LVEF (gt)'] = (g["dyastole volume ground truth"][...]-g["systole volume ground truth"][...])/(g["dyastole volume ground truth"][...]+1e-7)
			measurements['Dyastolic volume (gt)'] = g["dyastole volume ground truth"][...]
			measurements['Systolic volume (gt)'] = g["systole volume ground truth"][...]

		df = pd.DataFrame([measurements])
		df.to_csv(self.output().path, index=False)

	def output(self):
		try:
			os.makedirs(os.path.join(report_folder, self.dcm_name,self.__class__.__name__))
		except:
			pass

		return luigi.LocalTarget(os.path.join(report_folder, self.dcm_name,self.__class__.__name__, f"measurements_gt.csv"))

class MeasurementsStrainDCM(luigi.Task):
	dcm_name=luigi.Parameter()

	def requires(self):
		return {'strain':StrainDCM(dcm_name = self.dcm_name),
				'volumes':VolumeCurvesDCM(dcm_name = self.dcm_name),
				'info':InfoDCM(dcm_name = self.dcm_name)}

	def run(self):
		setupLog(self.__class__.__name__)

		f = h5py.File(self.input()['strain']['h5'].path, 'r')
		g = h5py.File(self.input()['volumes']['h5'].path, 'r')
		with open(self.input()['volumes']['dyas_sys_list'].path, 'r') as handle:
			lines = handle.read().splitlines()
			try:
				dyas_list = lines[0].split(',')
			except:
				dyas_list = []
			try:
				sys_list = lines[1].split(',')
			except:
				sys_list = []
			dyas_list = np.array(dyas_list, dtype=float).astype(int)
			sys_list = np.array(sys_list, dtype=float).astype(int)

		with open(self.input()['info']['pickle'].path, 'rb') as handle:
			info = pickle.load(handle)

		if info["fps"] is None:
			fps = 50
		else:
			fps = info["fps"]

		gls = f["gls_rate"][...].cumsum()
		rls = f["rls_rate"][...].cumsum(axis = 0)
		gts = f["gts_rate"][...].cumsum()
		rts = f["rts_rate"][...].cumsum(axis = 0)

		for dyas in dyas_list:
			if (dyas < gls.shape[0]):
				gls[dyas:] = gls[dyas:] - gls[dyas]
				gts[dyas:] = gts[dyas:] - gts[dyas]
				for region in range(6):
					rls[dyas:, region] = rls[dyas:, region] - rls[dyas, region]
					rts[dyas:, region] = rts[dyas:, region] - rts[dyas, region]

		measurements = {}
		measurements['gls_peak_contraction'] = gls[dyas_list[0]:].min()
		measurements['gls_peak_distention'] = gls[dyas_list[0]:].max()
		measurements['gls_peak_contraction_rate'] = f['gls_rate'][...].min()*fps
		measurements['gls_peak_distention_rate'] = f['gls_rate'][...].max()*fps
		for region in range(6):
			measurements[f'rls{region}_peak_contraction'] = rls[dyas_list[0]:,region].min()
			measurements[f'rls{region}_peak_distention'] = rls[dyas_list[0]:,region].max()
			measurements[f'rls{region}_peak_contraction_rate'] = f['rls_rate'][...,region].min()*fps
			measurements[f'rls{region}_peak_distention_rate'] = f['rls_rate'][...,region].max()*fps
		measurements['gts_peak_contraction'] = gts[dyas_list[0]:].min()
		measurements['gts_peak_distention'] = gts[dyas_list[0]:].max()
		measurements['gts_peak_contraction_rate'] = f['gts_rate'][...].min()*fps
		measurements['gts_peak_distention_rate'] = f['gts_rate'][...].max()*fps
		for region in range(6):
			measurements[f'rts{region}_peak_contraction'] = rls[dyas_list[0]:,region].min()
			measurements[f'rts{region}_peak_distention'] = rls[dyas_list[0]:,region].max()
			measurements[f'rts{region}_peak_contraction_rate'] = f['rts_rate'][...,region].min()*fps
			measurements[f'rts{region}_peak_distention_rate'] = f['rts_rate'][...,region].max()*fps

		df = pd.DataFrame([measurements])
		df.to_csv(self.output().path, index=False)

	def output(self):
		try:
			os.makedirs(os.path.join(report_folder, self.dcm_name,self.__class__.__name__))
		except:
			pass

		return luigi.LocalTarget(os.path.join(report_folder, self.dcm_name,self.__class__.__name__, f"measurements_strain.csv"))

class ConfidenceCheck(luigi.Task):
	dcm_name = luigi.Parameter()

	def requires(self):
		return {'segmentventricle':SegmentVentricleDCM(dcm_name = self.dcm_name),
				'gt':GTMasksAndVolumes(dcm_name = self.dcm_name),
				'volumes':VolumeCurvesDCM(dcm_name = self.dcm_name)}

	def run(self):

		f = h5py.File(self.input()["segmentventricle"].path, "r")
		g = h5py.File(self.input()['gt']['h5'].path,'r')
		h = h5py.File(self.input()['volumes']['h5'].path,'r')

		with open(self.input()['volumes']['dyas_sys_list'].path, 'r') as handle:
			lines = handle.read().splitlines()
			try:
				dyas_list = lines[0].split(',')
			except:
				dyas_list = []
			try:
				sys_list = lines[1].split(',')
			except:
				sys_list = []
			dyas_list = np.array(dyas_list, dtype=float).astype(int)
			sys_list = np.array(sys_list, dtype=float).astype(int)


		dice1 = h["dice avg+1 over avg-1"][...]
		dice2 = h["dice frame over frame"][...]
		dice3 = h["vol differences"][...]
		smoothed_curve = h["volume curve smoothed"][...]

		fig, ax = plt.subplots(figsize=(20,10))
		ax.fill_between(np.arange(h["volume curve avg"].shape[0]),h["volume curve avg+1"],h["volume curve avg-1"], alpha=0.5 )
		ax.plot(h["volume curve original"][...], c='black', alpha = 0.3)
		ax.plot(h["volume curve avg"][...], c='#1f77b4')
		# ax.plot(smoothed_curve, c='darkred')

		for dyas in dyas_list:
			plt.plot([dyas, dyas], [0.1*h["volume curve avg"][...].max(),1.1*h["volume curve avg"][...].max()], c='b')
		for sys in sys_list:
			plt.plot([sys, sys], [0.1*h["volume curve avg"][...].max(),1.1*h["volume curve avg"][...].max()], c='r')
		if "dyastole volume ground truth" in g.keys():
			plt.scatter(g["dyastole frame ground truth"][...], g["dyastole volume ground truth"][...], c='b', marker='x')
			plt.scatter(g["systole frame ground truth"][...], g["systole volume ground truth"][...], c='r', marker='x')

			max_vol = np.maximum(h["volume curve avg"][...].max(), g["dyastole volume ground truth"][...])
		else:
			max_vol = h["volume curve avg"][...].max()

		ax.set_ylim([0,1.2*max_vol])

		ax2 = ax.twinx()
		ax2.set_ylim([0,1])
		ax2.plot(dice1, color = 'orange')
		ax2.plot(dice2, color = 'green')
		ax2.plot(dice3, color = 'purple')

		ax2.plot([0, len(h["volume curve avg"][...])],[0.8,0.8], color = 'orange', linestyle='dashed', alpha = 0.5)
		ax2.plot([0, len(h["volume curve avg"][...])],[0.85,0.85], color = 'green', linestyle='dashed', alpha = 0.5)

		for i in range(len(dice1)):
			if ( h["reliable frames"][i]==0. ):
				rect = plt.Rectangle((i-0.5,0),1,1, color='red', alpha=0.1, linewidth=0)
				ax2.add_patch(rect)

		ax.set_title("Volume Curve and confidence dice")
		fig.savefig(self.output().path)


	def output(self):
		try:
			os.makedirs(os.path.join(report_folder, self.dcm_name,self.__class__.__name__))
		except:
			pass

		return luigi.LocalTarget(os.path.join(report_folder, self.dcm_name,self.__class__.__name__, f"confidence.png"))

class StrainCurves(luigi.Task):
	dcm_name = luigi.Parameter()

	def requires(self):
		return {'strain':StrainDCM(dcm_name = self.dcm_name),
				'track' :TrackDCM(dcm_name = self.dcm_name),
				'volumes':VolumeCurvesDCM(dcm_name = self.dcm_name)}

	def run(self):
		setupLog(self.__class__.__name__)

		try:
			os.makedirs(os.path.join(tmp_folder,self.__class__.__name__))
		except:
			pass

		file_path = os.path.join(dicom_folder,self.dcm_name+'.gtp')
		if os.path.isfile(file_path):
			ground_truth = True
			gt = h5py.File(file_path,'r')
		else:
			ground_truth = False

		f = h5py.File(self.input()['strain']['h5'].path,'r')
		g = h5py.File(self.input()['track'].path,'r')

		with open(self.input()['volumes']['dyas_sys_list'].path, 'r') as handle:
			lines = handle.read().splitlines()
			try:
				dyas_list = lines[0].split(',')
			except:
				dyas_list = []
			try:
				sys_list = lines[1].split(',')
			except:
				sys_list = []
			dyas_list = np.array(dyas_list, dtype=float).astype(int)
			sys_list = np.array(sys_list, dtype=float).astype(int)

		gls_rate = {}
		rls_rate = {}
		gls = {}
		rls = {}

		if ground_truth:
			algorithms = ['gt', 'ssd','strain_dl']
		else:
			algorithms = ['strain_dl']

		for alg in algorithms:
			gls_rate[alg] = np.zeros(g["tracking"].shape[0])
			rls_rate[alg] = np.zeros([g["tracking"].shape[0],6])

		if ground_truth:
			for frame in range(g["tracking"].shape[0]):
				new_x_gt = gt['speckles_gt'][0,:,0, frame+1]*gt['conversion_factor_x'][0]
				new_y_gt = gt['speckles_gt'][0,:,2, frame+1]*gt['conversion_factor_y'][0]
				old_x_gt = gt['speckles_gt'][0,:,0, frame]*gt['conversion_factor_x'][0]
				old_y_gt = gt['speckles_gt'][0,:,2, frame]*gt['conversion_factor_y'][0]

				gls_rate['gt'][frame],rls_rate['gt'][frame] = gls_rate_speckle(old_x_gt, old_y_gt, new_x_gt, new_y_gt)
				gls_rate['ssd'][frame],rls_rate['ssd'][frame] = gls_rate_tracking(old_x_gt, old_y_gt, g['tracking'][frame, :,:], self.requires()['track'].window_halfsize)

		(gls_rate['strain_dl'],rls_rate['strain_dl']) = (f['gls_rate'][...], f['rls_rate'][...])

		for alg in algorithms:
			gls[alg] = gls_rate[alg].cumsum()
			rls[alg] = rls_rate[alg].cumsum(axis = 0)
			for dyas in dyas_list[...]:
				if (dyas < gls[alg].shape[0]):
					gls[alg][dyas:] = gls[alg][dyas:] - gls[alg][dyas]
					# gts[alg][dyas:] = gts[alg][dyas:] - gts[alg][dyas]
					for region in range(6):
						rls[alg][dyas:, region] = rls[alg][dyas:, region] - rls[alg][dyas, region]
						# rts[alg][dyas:, region] = rts[alg][dyas:, region] - rts[alg][dyas, region]

		for region in range(6):
			plt.figure(figsize=(15,7))
			if ground_truth:
				plt.plot(rls['gt'][:,region], label='gt')
				plt.plot(rls['ssd'][:,region], label='ssd')
			plt.plot(rls['strain_dl'][:,region], label='strain_dl')
			for i in range(len(dyas_list)):
				dyas = dyas_list[i]
				plt.plot([dyas, dyas], [rls['strain_dl'][:,region].min(), rls['strain_dl'][:,region].max()], c='b')
			for i in range(len(sys_list)):
				sys = sys_list[i]
				plt.plot([sys, sys], [rls['strain_dl'][:,region].min(), rls['strain_dl'][:,region].max()], c='r')
			plt.legend()
			plt.title(f"rls_{region}")
			plt.savefig(self.output()[f"rls_{region}"].path)

		plt.figure(figsize=(15,7))
		if ground_truth:
			plt.plot(gls['gt'][:], label='gt')
			plt.plot(gls['ssd'][:], label='ssd')
		plt.plot(gls['strain_dl'][:], label='strain_dl')
		for i in range(len(dyas_list)):
			dyas = dyas_list[i]
			plt.plot([dyas, dyas], [gls['strain_dl'][:].min(), gls['strain_dl'][:].max()], c='b')
		for i in range(len(sys_list)):
			sys = sys_list[i]
			plt.plot([sys, sys], [gls['strain_dl'][:].min(), gls['strain_dl'][:].max()], c='r')
		plt.title("gls")
		plt.legend()
		plt.savefig(self.output()["gls"].path)

		curves_path = os.path.join(tmp_folder,self.__class__.__name__, f'curves_{self.dcm_name}.h5')
		curves_file = h5py.File(curves_path, 'w')

		if ground_truth:
			curves_file.create_dataset("gls_rate gt", (g["tracking"].shape[0]) , dtype = "float32")
			curves_file.create_dataset("rls_rate gt", (g["tracking"].shape[0],6) , dtype = "float32")
			curves_file.create_dataset("gls_rate ssd", (g["tracking"].shape[0]) , dtype = "float32")
			curves_file.create_dataset("rls_rate ssd", (g["tracking"].shape[0],6) , dtype = "float32")
		curves_file.create_dataset("gls_rate strain_dl", (g["tracking"].shape[0]) , dtype = "float32")
		curves_file.create_dataset("rls_rate strain_dl", (g["tracking"].shape[0],6) , dtype = "float32")

		if ground_truth:
			curves_file["gls_rate gt"][...] = gls_rate['gt']
			curves_file["rls_rate gt"][...] = rls_rate['gt']
			curves_file["gls_rate ssd"][...] = gls_rate['ssd']
			curves_file["rls_rate ssd"][...] = rls_rate['ssd']
		curves_file["gls_rate strain_dl"][...] = gls_rate['strain_dl']
		curves_file["rls_rate strain_dl"][...] = rls_rate['strain_dl']

		curves_file.close()

		os.rename(curves_path, self.output()['h5'].path)

	def output(self):
		try:
			os.makedirs(os.path.join(report_folder, self.dcm_name,self.__class__.__name__))
		except:
			pass

		dic = {}
		dic['gls'] = luigi.LocalTarget(os.path.join(report_folder, self.dcm_name,self.__class__.__name__, f"gls.png"))
		for i in range(6):
			dic[f'rls_{i}'] = luigi.LocalTarget(os.path.join(report_folder, self.dcm_name,self.__class__.__name__, f"rls_{i}.png"))
		dic['h5'] = luigi.LocalTarget(os.path.join(report_folder, self.dcm_name,self.__class__.__name__, f"curves.h5"))

		return dic

class VolumeGraphs(luigi.Task):
	dcm_name = luigi.Parameter()

	def requires(self):
		return {'volume curves' : VolumeCurvesDCM(dcm_name=self.dcm_name),
				'gt': GTMasksAndVolumes(dcm_name=self.dcm_name)}

	def run(self):
		setupLog(self.__class__.__name__)

		# with open(self.input()['info']['csv'].path, 'r') as handle:
		# 	lines = handle.read().splitlines()
		# 	keys = lines[0].split(',')
		# 	values = lines[1].split(',')
		# 	info = {}
		# 	for i in range(len(keys)):
		# 		key = keys[i]
		# 		value = values[i]
		# 		if key == '':
		# 			key = f'Unnamed{i}'
		# 		if value == '':
		# 			value = np.nan
		# 		else:
		# 			value = float(value)
		# 			value = int(value)
		# 		info[key] = value

		f = h5py.File(self.input()['volume curves']["h5"].path, "r")
		g = h5py.File(self.input()['gt']["h5"].path, "r")

		with open(self.input()['volume curves']['dyas_sys_list'].path, 'r') as handle:
			lines = handle.read().splitlines()
			try:
				dyas_list = lines[0].split(',')
			except:
				dyas_list = []
			try:
				sys_list = lines[1].split(',')
			except:
				sys_list = []
			dyas_list = np.array(dyas_list, dtype=float).astype(int)
			sys_list = np.array(sys_list, dtype=float).astype(int)


		fig, ax = plt.subplots(figsize=(20,10))
		plt.fill_between(np.arange(f["volume curve avg"].shape[0]),f["volume curve avg+1"],f["volume curve avg-1"], alpha=0.5 )
		plt.plot(f["volume curve smoothed"][...], c='black', alpha = 0.3)
		plt.plot(f["volume curve avg"][...], c='#1f77b4')
		for dyas in dyas_list:
			plt.plot([dyas, dyas], [0.1*f["volume curve avg"][...].max(),1.1*f["volume curve avg"][...].max()], c='b')
		for sys in sys_list:
			plt.plot([sys, sys], [0.1*f["volume curve avg"][...].max(),1.1*f["volume curve avg"][...].max()], c='r')
		if "dyastole volume ground truth" in g.keys():
			plt.scatter(g["dyastole frame ground truth"][...], g["dyastole volume ground truth"][...], c='b', marker='x')
			plt.scatter(g["systole frame ground truth"][...], g["systole volume ground truth"][...], c='r', marker='x')

			max_vol = np.maximum(f["volume curve avg"][...].max(), g["dyastole volume ground truth"][...])
		else:
			max_vol = f["volume curve avg"][...].max()
		plt.ylim([0,1.2*max_vol])

		reliable_frames = (f["reliable frames"][...] == 1)
		for i in range(len(reliable_frames)):
			if ( reliable_frames[i] == False):
				rect = plt.Rectangle((i-0.5,0),1,1.2*max_vol, color='red', alpha=0.1, linewidth=0)
				ax.add_patch(rect)

		plt.title("Volume Curve")
		plt.savefig(self.output()["png"].path)

	def output(self):

		try:
			os.makedirs(os.path.join(report_folder, self.dcm_name,self.__class__.__name__))
		except:
			pass
		return {"png": luigi.LocalTarget(os.path.join(report_folder, self.dcm_name,self.__class__.__name__, f"volumecurve.png"))}

class VolumeCurvesDCM(luigi.Task):
	dcm_name = luigi.Parameter()
	smoothing_window = luigi.IntParameter(default=2)

	def requires(self):
		return {"segmentpoints": SegmentPointsDCM(dcm_name = self.dcm_name),
				'segmentventricle':SegmentVentricleDCM(dcm_name = self.dcm_name),
				"info": InfoDCM(dcm_name = self.dcm_name)}

	def run(self):
		setupLog(self.__class__.__name__)

		try:
			os.makedirs(os.path.join(tmp_folder,self.__class__.__name__))
		except:
			pass

		with open(self.input()['info']['pickle'].path, 'rb') as handle:
			info = pickle.load(handle)

		img_path = os.path.join(dicom_folder,self.dcm_name+'.dcm')
		img = read_dicom(img_path)
		if (info["min x0"] is not None):
			img = img[:,info["min y0"]:info["max y1"], info["min x0"]:info["max x1"],...]
			x0 = info["min x0"]
			y0 = info["min y0"]
		else:
			x0 = 0
			y0 = 0
		img = img.astype(float)
		img = img-img.min()/(img.max()-img.min())

		f = h5py.File(self.input()['segmentpoints'].path, 'r')
		file_path = os.path.join(tmp_folder,self.__class__.__name__,f'volume_curves_{self.dcm_name}.h5')
		g = h5py.File(file_path, 'w')
		h = h5py.File(self.input()['segmentventricle'].path, 'r')



#TODO: REPORT SOMEWHERE IF PIXEL SPACING IS MISSING
		if info['pixel spacing x'] is not None:
			pixel_spacing = np.sqrt(info['pixel spacing x'] * info['pixel spacing y'])
		else:
			pixel_spacing =  100.

		shape = h['segmentation avg'][...].shape


		g.create_dataset("volume curve original", (shape[0]), dtype = "float32")
		g.create_dataset("volume curve avg", (shape[0]), dtype = "float32")
		g.create_dataset("volume curve avg+1", (shape[0]), dtype = "float32")
		g.create_dataset("volume curve avg-1", (shape[0]), dtype = "float32")
		g.create_dataset("volume curve smoothed", (shape[0]), dtype = "float32")
		g.create_dataset("dice avg+1 over avg-1", (shape[0]), dtype = "float32")
		g.create_dataset("dice frame over frame", (shape[0]), dtype = "float32")
		g.create_dataset("normalized variance sum", (shape[0]), dtype = "float32")
		g.create_dataset("vol differences", (shape[0]), dtype = "float32")
		g.create_dataset("reliable frames", (shape[0]), dtype = "int8")

		for modifier in ["original", "avg", "avg+1", "avg-1"]:
			for frame in range(shape[0]):
				vol_frame = 0
				thickness = f[f"{modifier} seg disk thickness"][frame]
				for disk in range(20):
					(x1,y1) = f[f"{modifier} seg disks"][frame, disk, 0,:]
					(x2,y2) = f[f"{modifier} seg disks"][frame, disk, 1,:]
					vol_frame += np.pi/4.*((x1-x2)**2+(y1-y2)**2)*thickness*pixel_spacing**3
				g[f"volume curve {modifier}"][frame] = vol_frame

		reliability_dice1 = []
		reliability_dice2 = []
		normalized_variance_sum = []

		for frame in range(h['segmentation avg'].shape[0]):
			dice = 2*(h['segmentation avg+1'][frame,...]*h['segmentation avg-1'][frame,...]).sum()/(h['segmentation avg+1'][frame,...]+h['segmentation avg-1'][frame,...]).sum()
			reliability_dice1.append(dice)
		reliability_dice1 = np.array(reliability_dice1)

		for frame in range(h['segmentation avg'].shape[0]-1):
			dice = 2*(h['segmentation avg'][frame,...]*h['segmentation avg'][frame+1,...]).sum()/(h['segmentation avg'][frame,...]+h['segmentation avg'][frame+1,...]).sum()
			reliability_dice2.append(dice)
		reliability_dice2 = np.minimum(np.array([1.]+reliability_dice2), np.array(reliability_dice2+[1.]))

		for frame in range(h['segmentation avg'].shape[0]):
			nvs = h['augmented segmentations'][...].std(axis = 0).sum(axis = (1,2,3))/(h['segmentation avg'][...].sum(axis = (1,2,3))+1e-7)
			normalized_variance_sum.append(nvs)
		normalized_variance_sum = np.array(normalized_variance_sum)


		g['dice avg+1 over avg-1'][...] = reliability_dice1
		g['dice frame over frame'][...] = reliability_dice2
		# g["normalized variance sum"][...] = normalized_variance_sum
		g['vol differences'][...] = 1.-(g["volume curve avg-1"][...]-g["volume curve avg+1"][...])/(g["volume curve avg+1"][...]+g["volume curve avg-1"][...])

		reliable_frames = (g['dice avg+1 over avg-1'][...] > 0.8) & (g['dice frame over frame'][...]  > 0.85) & (g['vol differences'][...] > 0.75)
		g['reliable frames'][...] = reliable_frames.astype(np.int8)

		# vol_max = g["volume curve avg"][reliable_frames].max()
		# vol_min = g["volume curve avg"][reliable_frames].min()



		smoothed_curve = np.zeros(shape[0], dtype="float32")
		n_frames_smoothed = np.zeros(shape[0], dtype="float32")

		if(reliable_frames.sum()>1):
			vol_curve_interpolated = g["volume curve avg"][...]
			vol_curve_interpolated[~reliable_frames] = np.interp(np.where(~reliable_frames)[0], np.where(reliable_frames)[0], vol_curve_interpolated[reliable_frames])

			for i in range(-self.smoothing_window, self.smoothing_window+1):
				smoothed_curve[max(0,i):min(shape[0],shape[0]+i)] += vol_curve_interpolated[max(0,-i):min(shape[0],shape[0]-i)]
				n_frames_smoothed[max(0,i):min(shape[0],shape[0]+i)] += np.ones(min(shape[0]-i, shape[0]+i))

			smoothed_curve = smoothed_curve/n_frames_smoothed

			sys_list, dyas_list = find_systole_dyastole(vol_curve_interpolated)

		else:
			#Not enough reliable frames, we leave empty the systole and dyastole frames
			dyas_list = []
			sys_list = []

			for i in range(-self.smoothing_window, self.smoothing_window+1):
				smoothed_curve[max(0,i):min(shape[0],shape[0]+i)] += g["volume curve avg"][max(0,-i):min(shape[0],shape[0]-i)]
				n_frames_smoothed[max(0,i):min(shape[0],shape[0]+i)] += np.ones(min(shape[0]-i, shape[0]+i))
			smoothed_curve = smoothed_curve/n_frames_smoothed

		g['volume curve smoothed'][...] = smoothed_curve

		g.create_dataset("dyastole list", data = np.array(dyas_list), dtype = "int32")
		g.create_dataset("systole list", data = np.array(sys_list), dtype = "int32")

		dyas_sys_list_path = os.path.join(tmp_folder,self.__class__.__name__,f'dyas_sys_list_{self.dcm_name}.txt')

		with open(dyas_sys_list_path, 'w') as handle:
			handle.write(f"{list(dyas_list)}"[1:-1])
			handle.write("\n")
			handle.write(f"{list(sys_list)}"[1:-1])

		f.close()
		g.close()

		#TODO renaming into existing filename...
		os.rename(file_path, self.output()['h5'].path)
		os.rename(dyas_sys_list_path, self.output()['dyas_sys_list'].path)

	def output(self):

		try:
			os.makedirs(os.path.join(report_folder, self.dcm_name,self.__class__.__name__))
		except:
			pass
		return {"dyas_sys_list": luigi.LocalTarget(os.path.join(report_folder, self.dcm_name,self.__class__.__name__, f"dyas_sys_list.txt")),
				"h5": luigi.LocalTarget(os.path.join(report_folder, self.dcm_name,self.__class__.__name__, f"volumecurve.h5"))}

class GTMasksAndVolumes(luigi.Task):
	dcm_name = luigi.Parameter()

	def requires(self):
		return {"segmentpoints": SegmentPointsDCM(dcm_name = self.dcm_name),
				'segmentventricle':SegmentVentricleDCM(dcm_name = self.dcm_name),
				"info": InfoDCM(dcm_name = self.dcm_name)}

	def run(self):
		setupLog(self.__class__.__name__)

		with open(self.input()['info']['pickle'].path, 'rb') as handle:
			info = pickle.load(handle)

		#TODO: REPORT SOMEWHERE IF PIXEL SPACING IS MISSING
		if info['pixel spacing x'] is not None:
			pixel_spacing = np.sqrt(info['pixel spacing x'] * info['pixel spacing y'])
		else:
			pixel_spacing =  100.

		try:
			os.makedirs(os.path.join(tmp_folder,self.__class__.__name__))
		except:
			pass

		img_path = os.path.join(dicom_folder,self.dcm_name+'.dcm')
		img = read_dicom(img_path)
		if (info["min x0"] is not None):
			img = img[:,info["min y0"]:info["max y1"], info["min x0"]:info["max x1"],...]
			x0 = info["min x0"]
			y0 = info["min y0"]
		else:
			x0 = 0
			y0 = 0
		img = img.astype(float)
		img = img-img.min()/(img.max()-img.min())

		f = h5py.File(self.input()['segmentpoints'].path, 'r')
		file_path = os.path.join(tmp_folder,self.__class__.__name__,f'volume_gt_{self.dcm_name}.h5')
		g = h5py.File(file_path, 'w')
		h = h5py.File(self.input()['segmentventricle'].path, 'r')

		vol_dyas = 0
		vol_sys = 0

		if os.path.isfile(os.path.join(dicom_folder,'masks.xlsx')):

			df = pd.read_excel(os.path.join(dicom_folder,'masks.xlsx'))
			if self.dcm_name in df['INSTANCEFILENAME'].str.strip().unique():
				try:
					gt = True

					#TODO: Si hay varias máscaras de diástole o sístole en el mismo frame, estoy cogiendo la unión de todas ellas
					#Probablement habría que corregir en la query SQL
					frameDyas = df.loc[(df['INSTANCEFILENAME'].str.strip()==self.dcm_name)&(df['MEDIDA'].str.strip().isin(['VTD VI (4C)','Area VI td 4C', 'VTD VI (2C)','Area VI td 2C'])), 'FRAME'].values[0]
					frameSys = df.loc[(df['INSTANCEFILENAME'].str.strip()==self.dcm_name)&(df['MEDIDA'].str.strip().isin(['VTS VI (4C)','Area VI ts 4C', 'VTS VI (2C)','Area VI ts 2C'])), 'FRAME'].values[0]
					mask_verticesDyas = df.loc[(df['INSTANCEFILENAME'].str.strip()==self.dcm_name)&(df['FRAME']==frameDyas),['NATIVEX','NATIVEY']].values-np.array([x0,y0])
					mask_verticesSys = df.loc[(df['INSTANCEFILENAME'].str.strip()==self.dcm_name)&(df['FRAME']==frameSys),['NATIVEX','NATIVEY']].values-np.array([x0,y0])
					mask_arrayDyas = np.zeros([*(img.shape[1:3]),1])
					mask_arrayDyas = cv2.fillConvexPoly(mask_arrayDyas, mask_verticesDyas,1)[...,0]
					mask_arraySys = np.zeros([*(img.shape[1:3]),1])
					mask_arraySys = cv2.fillConvexPoly(mask_arraySys, mask_verticesSys,1)[...,0]

					(x_mpc,y_mpc), (x_mp1, y_mp1), (x_mp2, y_mp2), (x_ap, y_ap) = find_mitral_plane_and_apex_1mask(mask_arrayDyas)
					coord_list = find_disks_1mask(mask_arrayDyas, x_mpc, y_mpc, x_ap, y_ap)
					disk_thicknessDyas = np.sqrt((x_ap-x_mpc)**2+(y_ap-y_mpc)**2)/20.

					for (x1,y1,x2,y2) in coord_list:
						vol_dyas += np.pi/4.*((x1-x2)**2+(y1-y2)**2)*disk_thicknessDyas*pixel_spacing**3

					(x_mpc,y_mpc), (x_mp1, y_mp1), (x_mp2, y_mp2), (x_ap, y_ap) = find_mitral_plane_and_apex_1mask(mask_arraySys)
					coord_list = find_disks_1mask(mask_arraySys, x_mpc, y_mpc, x_ap, y_ap)
					disk_thicknessSys = np.sqrt((x_ap-x_mpc)**2+(y_ap-y_mpc)**2)/20.

					for (x1,y1,x2,y2) in coord_list:
						vol_sys += np.pi/4.*((x1-x2)**2+(y1-y2)**2)*disk_thicknessSys*pixel_spacing**3

					img_array = (img[frameDyas,...,0]-img[frameDyas,...,0].min())/(img[frameDyas,...,0].max()-img[frameDyas,...,0].min())
					plt.figure(figsize = (10,10))
					plt.imshow(img_array, cmap = 'gray')
					plt.imshow(h['segmentation avg'][frameDyas,...,0], alpha=0.5)
					plt.savefig(os.path.join(report_folder, self.dcm_name,self.__class__.__name__, f"Dyas_Seg.png"))
					plt.figure(figsize = (10,10))
					plt.imshow(img_array, cmap = 'gray')
					plt.imshow(mask_arrayDyas, alpha=0.5)
					plt.savefig(os.path.join(report_folder, self.dcm_name,self.__class__.__name__, f"Dyas_GT.png"))
					img_array = (img[frameSys,...,0]-img[frameSys,...,0].min())/(img[frameSys,...,0].max()-img[frameSys,...,0].min())
					plt.figure(figsize = (10,10))
					plt.imshow(img_array, cmap = 'gray')
					plt.imshow(h['segmentation avg'][frameSys,...,0], alpha=0.5)
					plt.savefig(os.path.join(report_folder, self.dcm_name,self.__class__.__name__, f"Sys_Seg.png"))
					plt.figure(figsize = (10,10))
					plt.imshow(img_array, cmap = 'gray')
					plt.imshow(mask_arraySys, alpha=0.5)
					plt.savefig(os.path.join(report_folder, self.dcm_name,self.__class__.__name__, f"Sys_GT.png"))
				except Exception as e:
					try:
						with open(os.path.join(report_folder, self.dcm_name,self.__class__.__name__, f"MasksError.txt"), "w") as f:
							f.write(repr(e))
					except:
						pass
					gt=False
			else:
				gt = False
		else:
			gt = False

		if gt:
			g.create_dataset("dyastole volume ground truth", data = np.array(vol_dyas), dtype = "float32")
			g.create_dataset("systole volume ground truth", data = np.array(vol_sys), dtype = "float32")
			g.create_dataset("dyastole frame ground truth", data = np.array(frameDyas), dtype = "float32")
			g.create_dataset("systole frame ground truth", data = np.array(frameSys), dtype = "float32")

		g.close()

		#TODO renaming into existing filename...
		os.rename(file_path, self.output()['h5'].path)

	def output(self):
		try:
			os.makedirs(os.path.join(report_folder, self.dcm_name,self.__class__.__name__))
		except:
			pass
		return {"h5": luigi.LocalTarget(os.path.join(report_folder, self.dcm_name,self.__class__.__name__, f"volume_gt.h5"))}


class VideoSegmentationDCM(luigi.Task):
	dcm_name = luigi.Parameter()
	myo_segmentation = luigi.BoolParameter(default = False)

	def requires(self):
		requirements = {"segmentation": SegmentVentricleDCM(dcm_name = self.dcm_name),
						"segmentpoints": SegmentPointsDCM(dcm_name = self.dcm_name),
						"volumes": VolumeCurvesDCM(dcm_name = self.dcm_name),
						"gt": GTMasksAndVolumes(dcm_name = self.dcm_name),
						"info": InfoDCM(dcm_name = self.dcm_name)}
		if self.myo_segmentation:
			requirements["myocardium"] = MyocardiumRegionsDCM(dcm_name = self.dcm_name)
		return requirements

	def run(self):
		setupLog(self.__class__.__name__)

		with open(self.input()['info']['pickle'].path, 'rb') as handle:
			info = pickle.load(handle)

		file_path = os.path.join(dicom_folder,self.dcm_name+'.dcm')
		img = read_dicom(file_path)

		if (info["min x0"] is not None):
			img = img[:,info["min y0"]:info["max y1"], info["min x0"]:info["max x1"],...]

		img = img.astype(float)
		img = img-img.min()/(img.max()-img.min())

		f = h5py.File(self.input()["volumes"]["h5"].path, "r")
		g = h5py.File(self.input()["gt"]["h5"].path, "r")
		h = h5py.File(self.input()["segmentpoints"].path, "r")
		s = h5py.File(self.input()["segmentation"].path, "r")
		if self.myo_segmentation:
			m = h5py.File(self.input()["myocardium"].path, "r")


		with open(self.input()["volumes"]['dyas_sys_list'].path, 'r') as handle:
			lines = handle.read().splitlines()
			try:
				dyas_list = lines[0].split(',')
			except:
				dyas_list = []
			try:
				sys_list = lines[1].split(',')
			except:
				sys_list = []
			dyas_list = np.array(dyas_list, dtype=float).astype(int)
			sys_list = np.array(sys_list, dtype=float).astype(int)


		if info["fps"] is None:
			fps = 50
		else:
			fps = info["fps"]

		if info["pixel spacing x"] is None:
			pixel_spacing = 100
		else:
			pixel_spacing = np.sqrt(info["pixel spacing x"]*info["pixel spacing y"])

		if "dyastole volume ground truth" in g.keys():
			gt = True
			vol_dyas = g["dyastole volume ground truth"][...]
			vol_sys = g["systole volume ground truth"][...]
			frameDyas = g["dyastole frame ground truth"][...]
			frameSys = g["systole frame ground truth"][...]
		else:
			gt = False

		height = img.shape[1]
		width = img.shape[2]

		path = os.path.join(tmp_folder, self.__class__.__name__, self.dcm_name+f'_tmpvideo{"_myo" if self.myo_segmentation else ""}.mp4')
		try:
			os.makedirs(os.path.join(tmp_folder,self.__class__.__name__))
		except:
			pass
		writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'H264'), fps/4., (2*width,2*height))
		frame_array = np.zeros([2*height, 2*width,3], dtype='uint8')
		for frame in range(img.shape[0]):
			img_array = (img[frame,...,0]-img[...,0].min())/(img[...,0].max()-img[...,0].min())
			frame_array[0:height, 0:width,:] = 255*np.tensordot(img_array, np.ones(3), axes=0)

			overlay_array = np.zeros([height,width,3])
			if self.myo_segmentation:
				overlay_array[...,0] = 255*((m["avg myo seg"][frame,...,0]==0)*img_array+(m["avg myo seg"][frame,...,0]==1)*(0.9*img_array+0.1))
				overlay_array[...,1] = 255*((m["avg myo seg"][frame,...,0]==0)*img_array+(m["avg myo seg"][frame,...,0]==2)*(0.9*img_array+0.1))
				overlay_array[...,2] = 255*((m["avg myo seg"][frame,...,0]==0)*img_array+(m["avg myo seg"][frame,...,0]==3)*(0.9*img_array+0.1))
			else:
				overlay_array[...,0] = 255*((s["segmentation avg"][frame,...,0]==0)*img_array+(s["segmentation avg"][frame,...,0]==1)*(0.7*img_array+0.3))
				overlay_array[...,1] = 255*((s["segmentation avg"][frame,...,0]==0)*img_array+(s["segmentation avg"][frame,...,0]==1)*(0.7*img_array+0.3))
				overlay_array[...,2] = 255*(s["segmentation avg"][frame,...,0]==0)*img_array
			frame_array[0:height, width:2*width,:] = overlay_array

			avg_segmentation_array = np.zeros([height,width,3])
			avg_segmentation_array[...,0] = 255*(s["segmentation avg"][frame,...,0]==1).astype('uint8')
			if self.myo_segmentation:
				avg_segmentation_array[...,1] = 255*(m["avg myo seg"][frame,...,0]==2).astype('uint8')
				avg_segmentation_array[...,2] = 255*(m["avg myo seg"][frame,...,0]==3).astype('uint8')
			else:
				avg_segmentation_array[...,1] = 255*(s["segmentation avg"][frame,...,0]==1).astype('uint8')

			for disk in range(20):
				print(frame, disk, h["avg seg disks"])
				(x1,y1) = h["avg seg disks"][frame, disk, 0,:]
				(x2,y2) = h["avg seg disks"][frame, disk, 1,:]
				yl_list, xl_list = skimage.draw.line(np.rint(y1).astype(int), np.rint(x1).astype(int), np.rint(y2).astype(int),np.rint(x2).astype(int))
				avg_segmentation_array[yl_list, xl_list,:] = 255
			[x_ap,y_ap] = h["avg seg points"][frame,0,:]
			[x_mpc,y_mpc] = h["avg seg points"][frame,1,:]
			[x_mp1,y_mp1] = h["avg seg points"][frame,2,:]
			[x_mp2,y_mp2] = h["avg seg points"][frame,3,:]
			yl_list, xl_list = skimage.draw.line(np.rint(y_ap).astype(int), np.rint(x_ap).astype(int), np.rint(y_mpc).astype(int),np.rint(x_mpc).astype(int))
			avg_segmentation_array[yl_list, xl_list,:] = 255
			yl_list, xl_list = skimage.draw.disk((y_mp1, x_mp1), 3)
			avg_segmentation_array[yl_list, xl_list,0] = 0
			avg_segmentation_array[yl_list, xl_list,1] = 128
			avg_segmentation_array[yl_list, xl_list,2] = 255
			yl_list, xl_list = skimage.draw.disk((y_mp2, x_mp2), 3)
			avg_segmentation_array[yl_list, xl_list,0] = 0
			avg_segmentation_array[yl_list, xl_list,1] = 128
			avg_segmentation_array[yl_list, xl_list,2] = 255
			yl_list, xl_list = skimage.draw.disk((y_mpc, x_mpc), 3)
			avg_segmentation_array[yl_list, xl_list,0] = 0
			avg_segmentation_array[yl_list, xl_list,1] = 255
			avg_segmentation_array[yl_list, xl_list,2] = 255
			yl_list, xl_list = skimage.draw.disk((y_ap, x_ap), 3)
			avg_segmentation_array[yl_list, xl_list,0] = 0
			avg_segmentation_array[yl_list, xl_list,1] = 0
			avg_segmentation_array[yl_list, xl_list,2] = 255
			frame_array[height:2*height, 0:width,:] = avg_segmentation_array

			dpi = 100
			sidex = width / dpi
			sidey = height / dpi
			fig = plt.figure(figsize=(sidex,sidey), dpi=dpi)
			ax = fig.gca()

			ax.plot(f["volume curve smoothed"][...], color = 'black', alpha = 0.3)
			ax.plot(f["volume curve avg"][...])

			for i in range(len(dyas_list)):
				dyas = dyas_list[i]
				ax.plot([dyas, dyas], [0.1*f["volume curve avg"][...].max(),1.1*f["volume curve avg"][...].max()], c='b')
			for i in range(len(sys_list)):
				sys = sys_list[i]
				ax.plot([sys, sys], [0.1*f["volume curve avg"][...].max(),1.1*f["volume curve avg"][...].max()], c='r')
			ax.plot([frame, frame], [0.1*f["volume curve avg"][...].max(),1.1*f["volume curve avg"][...].max()], c='darkgreen')

			if gt:
				ax.scatter(frameSys, vol_sys, c='r', marker='x')
				ax.scatter(frameDyas, vol_dyas, c='b', marker='x')
				max_vol = np.maximum(f["volume curve avg"][...].max(), vol_dyas)
			else:
				max_vol = f["volume curve avg"][...].max()

			ax.set_ylim([0,1.2*max_vol])

			reliable_frames = (f["reliable frames"][...] == 1)
			for i in range(len(reliable_frames)):
				if ( reliable_frames[i] == False):
					rect = plt.Rectangle((i-0.5,0),1,1.2*max_vol, color='red', alpha=0.1, linewidth=0)
					ax.add_patch(rect)

			io_buf = io.BytesIO()
			fig.savefig(io_buf, format='raw', dpi=dpi)
			io_buf.seek(0)
			img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
								 newshape=(int(sidey*dpi), int(sidex*dpi), -1))
			io_buf.close()
			frame_array[height:2*height, width:2*width,:] = img_arr[...,2::-1]

			writer.write(frame_array.astype('uint8'))
		writer.release()
		os.rename(path, self.output().path)

		f.close()

	def output(self):
		try:
			os.makedirs(os.path.join(report_folder, self.dcm_name,self.__class__.__name__))
		except:
			pass
		if self.myo_segmentation:
			return luigi.LocalTarget(os.path.join(report_folder, self.dcm_name,self.__class__.__name__, f"segmentation_myo.mp4"))
		else:
			return luigi.LocalTarget(os.path.join(report_folder, self.dcm_name,self.__class__.__name__, f"segmentation.mp4"))

class VideoTrackingDCM(luigi.Task):
	dcm_name = luigi.Parameter()

	def requires(self):
		return {"track":TrackDCM(dcm_name = self.dcm_name),
				"segmentventricle":SegmentVentricleDCM(dcm_name = self.dcm_name),
				"myosegmentation":MyocardiumRegionsDCM(dcm_name = self.dcm_name),
				"strain": StrainDCM(dcm_name = self.dcm_name),
				"info": InfoDCM(dcm_name = self.dcm_name)}

	def run(self):
		setupLog(self.__class__.__name__)

		file_path = os.path.join(dicom_folder,self.dcm_name+'.dcm')
		img = read_dicom(file_path)
		img = img.astype(float)

		f = h5py.File(self.input()["strain"]["h5"].path, "r")
		g = h5py.File(self.input()["segmentventricle"].path, "r")
		m = h5py.File(self.input()["myosegmentation"].path, "r")

		with open(self.input()['info']['pickle'].path, 'rb') as handle:
			info = pickle.load(handle)
		if info["fps"] is None:
			fps = 50
		else:
			fps = info["fps"]

		if (info["min x0"] is not None):
			img = img[:,info["min y0"]:info["max y1"], info["min x0"]:info["max x1"],...]

		bbox_x0 = g["bounding box top left"][1]
		bbox_x1 = g["bounding box bottom right"][1]
		bbox_y0 = g["bounding box top left"][0]
		bbox_y1 = g["bounding box bottom right"][0]


		height = img.shape[1]
		width = img.shape[2]

		path = os.path.join(tmp_folder, self.__class__.__name__, self.dcm_name+'_tmpvideo.mp4')
		try:
			os.makedirs(os.path.join(tmp_folder,self.__class__.__name__))
		except:
			pass

		writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'H264'), fps/4., (2*width, 2*height))
		frame_array = np.zeros([2*height, 2*width,3])
		for frame in range(img.shape[0]-1):
			img_array = (img[frame,...,0]-img[frame,...,0].min())/(img[frame,...,0].max()-img[frame,...,0].min())
			frame_array[0:height, 0:width,:] = 255*np.tensordot(img_array, np.ones(3), axes=0)

			pred = m["avg myo seg"][frame,...,0]
			myo = np.tensordot((pred == 2).astype(int), np.ones(3), axes=0)

			strain_array = np.zeros([height,width,3])
			longitudinal_strain_field = np.zeros([height,width])
			longitudinal_strain_field[bbox_y0:bbox_y1,bbox_x0:bbox_x1] = f["longitudinal_strain_field"][frame,...]
			strain_array[...,0] = 255*np.maximum(0, 1+(np.minimum(longitudinal_strain_field,0)/0.05))
			strain_array[...,1] = 255*np.maximum(0, (1-np.abs(longitudinal_strain_field)/0.05))
			strain_array[...,2] = 255*np.maximum(0, 1-(np.maximum(longitudinal_strain_field,0))/0.05)
			frame_array[0:height, width:2*width,:] = myo*strain_array+(1-myo)*255*np.tensordot(img_array, np.ones(3), axes=0)

			region_list = m["avg seg myo regions"][frame,...]
			for i in range(6):
				region = region_list[...,i]
				perimeter = np.logical_xor(region,morphology.erosion(region))
				strain_array[...,0] = strain_array[...,0]*(1-perimeter)
				strain_array[...,1] = strain_array[...,1]*(1-perimeter)
				strain_array[...,2] = strain_array[...,2]*(1-perimeter)

			frame_array[height:2*height, 0:width,:] = strain_array

			# ori_segmentation_array = np.zeros([h,w,3])
			# ori_segmentation_array[...,0] = (g["original segmentation"][frame,...]==1).astype('uint8')
			# ori_segmentation_array[...,1] = (g["original segmentation"][frame,...]==2).astype('uint8')
			# ori_segmentation_array[...,2] = (g["original segmentation"][frame,...]==3).astype('uint8')
			# frame_array[h:2*h, w:2*w,:] = ori_segmentation_array

			writer.write(frame_array.astype('uint8'))
		writer.release()
		os.rename(path, self.output().path)

		f.close()
		g.close()

	def output(self):
		try:
			os.makedirs(os.path.join(report_folder, self.dcm_name,self.__class__.__name__))
		except:
			pass
		return luigi.LocalTarget(os.path.join(report_folder, self.dcm_name,self.__class__.__name__, f"strain_field.mp4"))

class VideoTrackingGT(luigi.Task):
	dcm_name = luigi.Parameter()

	def requires(self):
		return {"track":TrackDCM(dcm_name = self.dcm_name),
				"segmentation":SegmentVentricleDCM(dcm_name = self.dcm_name),
				"info":InfoDCM(dcm_name = self.dcm_name)}

	def run(self):
		setupLog(self.__class__.__name__)

		file_path = os.path.join(dicom_folder,self.dcm_name+'.gtp')

		gt = h5py.File(file_path,'r')

		file_path = os.path.join(dicom_folder,self.dcm_name+'.dcm')
		img = read_dicom(file_path)
		img = img.astype(float)

		g = h5py.File(self.input()["segmentation"].path, "r")
		tr = h5py.File(self.input()["track"].path, "r")

		with open(self.input()['info']['pickle'].path, 'rb') as handle:
			info = pickle.load(handle)
		if info["fps"] is None:
			fps = 50
		else:
			fps = info["fps"]


		window_halfsize = self.requires()["track"].window_halfsize

		h = img.shape[1]
		w = img.shape[2]

		path = os.path.join(tmp_folder, self.__class__.__name__, self.dcm_name+'_tmpvideo.mp4')
		try:
			os.makedirs(os.path.join(tmp_folder,self.__class__.__name__))
		except:
			pass

		writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'H264'), fps/4., (2*w, 2*h))
		frame_array = np.zeros([2*h, 2*w,3])
		for frame in range(img.shape[0]-1):

			img_array = (img[frame,...,0]-img[frame,...,0].min())/(img[frame,...,0].max()-img[frame,...,0].min())

			bwr = plt.get_cmap('bwr')
			plasma = plt.get_cmap('plasma')
			prgn = plt.get_cmap('PRGn')

			new_x_gt = gt['speckles_gt'][0,:,0, frame+1]*gt['conversion_factor_x'][0]
			new_y_gt = gt['speckles_gt'][0,:,2, frame+1]*gt['conversion_factor_y'][0]
			old_x_gt = gt['speckles_gt'][0,:,0, frame]*gt['conversion_factor_x'][0]
			old_y_gt = gt['speckles_gt'][0,:,2, frame]*gt['conversion_factor_y'][0]

			tracking = tr['tracking'][frame, ...]
			fx =  scipy.interpolate.interp2d(np.arange(tracking.shape[1]),np.arange(tracking.shape[0]), tracking[:,:,0]-window_halfsize)
			fy =  scipy.interpolate.interp2d(np.arange(tracking.shape[1]),np.arange(tracking.shape[0]), tracking[:,:,1]-window_halfsize)

			inc_x_tr = []
			inc_y_tr = []

			for (x,y) in zip(old_x_gt,old_y_gt):
				inc_x_tr.append(fx(x,y)[0])
				inc_y_tr.append(fy(x,y)[0])

			inc_x_tr = np.array(inc_x_tr)
			inc_y_tr = np.array(inc_y_tr)
			inc_x_gt = new_x_gt-old_x_gt
			inc_y_gt = new_y_gt-old_y_gt

			old_total_length = 0
			new_total_length = 0

			old_segment_lengths = np.zeros([5,35])
			new_segment_lengths_gt = np.zeros([5,35])
			new_segment_lengths_tr = np.zeros([5,35])

			dpi = 100
			sidex = w / dpi
			sidey = h / dpi

			fig1 = plt.figure(figsize=(sidex,sidey))
			ax1 = fig1.gca()
			fig2 = plt.figure(figsize=(sidex,sidey))
			ax2 = fig2.gca()
			fig3 = plt.figure(figsize=(sidex,sidey))
			ax3 = fig3.gca()
			fig4 = plt.figure(figsize=(sidex,sidey))
			ax4 = fig4.gca()

			ax1.imshow(img_array, cmap='gray')
			ax2.imshow(img_array, cmap='gray')
			ax3.imshow(img_array, cmap='gray')
			ax4.imshow(img_array, cmap='gray')

			ax3.quiver(old_x_gt, old_y_gt, inc_x_tr, -inc_y_tr, scale=5, scale_units='inches', linewidth=1, color=plasma(np.sqrt((inc_x_tr-inc_x_gt)**2+(inc_y_tr-inc_y_gt)**2)/3.))
			# ax4.quiver(old_x_gt, old_y_gt, inc_x_gt, -inc_y_gt, scale=5, scale_units='inches', linewidth=1, color=[0,0,0.5])

			for depth in range(5):
				old_x = old_x_gt[36*depth:36*(depth+1)]
				old_y = old_y_gt[36*depth:36*(depth+1)]

				new_x = new_x_gt[36*depth:36*(depth+1)]
				new_y = new_y_gt[36*depth:36*(depth+1)]

				inc_x = inc_x_tr[36*depth:36*(depth+1)]
				inc_y = inc_y_tr[36*depth:36*(depth+1)]

				old_segment_lengths[depth] = np.sqrt((old_x[1:]-old_x[:-1])**2+(old_y[1:]-old_y[:-1])**2)
				new_segment_lengths_gt[depth] = np.sqrt((new_x[1:]-new_x[:-1])**2+(new_y[1:]-new_y[:-1])**2)
				new_segment_lengths_tr[depth] = np.sqrt((old_x[1:]+inc_x[1:]-old_x[:-1]-inc_x[:-1])**2+
														(old_y[1:]+inc_y[1:]-old_y[:-1]-inc_y[:-1])**2)

				for i in range(len(old_x)-1):
					strain1 = (new_segment_lengths_tr[depth][i]-old_segment_lengths[depth][i])/old_segment_lengths[depth][i]
					color_strain = bwr(0.5+strain1/0.025)
					ax1.plot(old_x[i:(i+2)],old_y[i:(i+2)],c = color_strain)
					strain2 = (new_segment_lengths_gt[depth][i]-old_segment_lengths[depth][i])/old_segment_lengths[depth][i]
					color_strain = bwr(0.5+strain2/0.025)
					ax2.plot(old_x[i:(i+2)],old_y[i:(i+2)],c = color_strain)
					color_strain = prgn(0.5+(strain2-strain1)/0.0125)
					ax4.plot(old_x[i:(i+2)],old_y[i:(i+2)],c = color_strain)


			ax1.set_title('speckle tracking')
			ax2.set_title('ground truth')

			io_buf = io.BytesIO()
			fig1.savefig(io_buf, format='raw', dpi=dpi)
			io_buf.seek(0)
			img_arr1 = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
								 newshape=(int(sidey*dpi), int(sidex*dpi), -1))
			io_buf.close()

			io_buf = io.BytesIO()
			fig2.savefig(io_buf, format='raw', dpi=dpi)
			io_buf.seek(0)
			img_arr2 = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
								 newshape=(int(sidey*dpi), int(sidex*dpi), -1))
			io_buf.close()

			io_buf = io.BytesIO()
			fig3.savefig(io_buf, format='raw', dpi=dpi)
			io_buf.seek(0)
			img_arr3 = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
								 newshape=(int(sidey*dpi), int(sidex*dpi), -1))
			io_buf.close()

			io_buf = io.BytesIO()
			fig4.savefig(io_buf, format='raw', dpi=dpi)
			io_buf.seek(0)
			img_arr4 = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
								 newshape=(int(sidey*dpi), int(sidex*dpi), -1))
			io_buf.close()

			frame_array[0:h, 0:w,:] = img_arr1[...,2::-1]
			frame_array[0:h, w:2*w,:] = img_arr2[...,2::-1]
			frame_array[h:2*h, 0:w,:] = img_arr3[...,2::-1]
			frame_array[h:2*h, w:2*w,:] = img_arr4[...,2::-1]

			# ori_segmentation_array = np.zeros([h,w,3])
			# ori_segmentation_array[...,0] = (g["original segmentation"][frame,...]==1).astype('uint8')
			# ori_segmentation_array[...,1] = (g["original segmentation"][frame,...]==2).astype('uint8')
			# ori_segmentation_array[...,2] = (g["original segmentation"][frame,...]==3).astype('uint8')
			# frame_array[h:2*h, w:2*w,:] = ori_segmentation_array
			im_path = os.path.join(report_folder, self.dcm_name,self.__class__.__name__, f"frame_{frame}.png")
			cv2.imwrite(im_path, frame_array)
			writer.write(frame_array.astype('uint8'))
		writer.release()
		os.rename(path, self.output().path)

		g.close()
		tr.close()
		gt.close()

	def output(self):
		try:
			os.makedirs(os.path.join(report_folder, self.dcm_name,self.__class__.__name__))
		except:
			pass
		return luigi.LocalTarget(os.path.join(report_folder, self.dcm_name,self.__class__.__name__, f"tracking_gt.mp4"))

class StrainDCM(luigi.Task):
	dcm_name = luigi.Parameter()

	def requires(self):
		return {"track":TrackDCM(dcm_name = self.dcm_name),
				# "segmentpoints":SegmentPointsDCM(dcm_name = self.dcm_name),
				"segmentventricle":SegmentVentricleDCM(dcm_name = self.dcm_name),
				"myocardiumregions":MyocardiumRegionsDCM(dcm_name = self.dcm_name),
				"volumes":VolumeCurvesDCM(dcm_name = self.dcm_name)}

	def run(self):
		setupLog(self.__class__.__name__)
		f = h5py.File(self.input()["track"].path, "r")
		g = h5py.File(self.input()["myocardiumregions"].path, "r")
		m = h5py.File(self.input()["segmentventricle"].path, "r")
		v = h5py.File(self.input()["volumes"]["h5"].path, "r")

		shape = f["tracking"].shape
		h5_path = os.path.join(tmp_folder,self.__class__.__name__,self.dcm_name+'_strain.h5')
		try:
			os.makedirs(os.path.join(tmp_folder,self.__class__.__name__))
		except:
			pass
		h = h5py.File(h5_path, "w")
		h.create_dataset("longitudinal_strain_field", (shape[0],shape[1],shape[2]) , chunks = (4, shape[1],shape[2]), dtype = "float32")
		h.create_dataset("gls_rate", (shape[0]) , dtype = "float32")
		h.create_dataset("rls_rate", (shape[0],6) , dtype = "float32")
		h.create_dataset("transversal_strain_field", (shape[0],shape[1],shape[2]) , chunks = (4, shape[1],shape[2]), dtype = "float32")
		h.create_dataset("gts_rate", (shape[0]) , dtype = "float32")
		h.create_dataset("rts_rate", (shape[0],6) , dtype = "float32")

		for frame in range(f['tracking'].shape[0]):
			region_list = g["avg seg myo regions"][frame,
													m["bounding box top left"][0]:m["bounding box bottom right"][0],
													m["bounding box top left"][1]:m["bounding box bottom right"][1],
													:]
			lmask = longitudinal_mask(m["segmentation avg"][frame,
																m["bounding box top left"][0]:m["bounding box bottom right"][0],
																m["bounding box top left"][1]:m["bounding box bottom right"][1],
																0])
			pmask = np.zeros_like(lmask)
			pmask[...,0] = lmask[...,1]
			pmask[...,1] = -lmask[...,0]
			longitudinal_strain_field, gls_rate, rls_rate = strain_dl(f["tracking"][frame,...], lmask, region_list)
			#TODO: Este "transversal strain field" no se si tiene mucho sentido, porque asumimos que las
			#que se mantiene la perpendicularidad entre fibras
			transversal_strain_field, gts_rate, rts_rate = strain_dl(f["tracking"][frame,...], pmask, region_list)

			h["longitudinal_strain_field"][frame,...] = longitudinal_strain_field
			h["gls_rate"][frame] = gls_rate
			h["rls_rate"][frame,:] = rls_rate
			h["transversal_strain_field"][frame,...] = transversal_strain_field
			h["gts_rate"][frame] = gts_rate
			h["rts_rate"][frame,:] = rts_rate


		gls = h["gls_rate"][...].cumsum()
		rls = h["rls_rate"][...].cumsum(axis = 0)
		gts = h["gts_rate"][...].cumsum()
		rts = h["rts_rate"][...].cumsum(axis = 0)

		for dyas in v["dyastole list"][...]:
			if (dyas < gls.shape[0]):
				gls[dyas:] = gls[dyas:] - gls[dyas]
				gts[dyas:] = gts[dyas:] - gts[dyas]
				for region in range(6):
					rls[dyas:, region] = rls[dyas:, region] - rls[dyas, region]
					rts[dyas:, region] = rts[dyas:, region] - rts[dyas, region]

		# for region in range(6):
		# 	plt.figure(figsize=(15,7))
		# 	plt.plot(rls[:,region], label='strain_dl')
		# 	for i in range(v["dyastole list"].shape[0]):
		# 		dyas = v["dyastole list"][i]
		# 		plt.plot([dyas, dyas], [rls[:,region].min(), rls[:,region].max()], c='b')
		# 	for i in range(v["systole list"].shape[0]):
		# 		sys = v["systole list"][i]
		# 		plt.plot([sys, sys], [rls[:,region].min(), rls[:,region].max()], c='r')
		# 	plt.legend()
		# 	plt.title(f"rls_{region}")
		# 	plt.savefig(self.output()[f"rls_{region}"].path)
		#
		# plt.figure(figsize=(15,7))
		# plt.plot(gls[:], label='strain_dl')
		# for i in range(v["dyastole list"].shape[0]):
		# 	dyas = v["dyastole list"][i]
		# 	plt.plot([dyas, dyas], [gls.min(), gls.max()], c='b')
		# for i in range(v["systole list"].shape[0]):
		# 	sys = v["systole list"][i]
		# 	plt.plot([sys, sys], [gls.min(), gls.max()], c='r')
		# plt.title("gls")
		# plt.legend()
		# plt.savefig(self.output()["gls"].path)
		#
		# for region in range(6):
		# 	plt.figure(figsize=(15,7))
		# 	plt.plot(rts[:,region], label='strain_dl')
		# 	for i in range(v["dyastole list"].shape[0]):
		# 		dyas = v["dyastole list"][i]
		# 		plt.plot([dyas, dyas], [rts[:,region].min(), rts[:,region].max()], c='b')
		# 	for i in range(v["systole list"].shape[0]):
		# 		sys = v["systole list"][i]
		# 		plt.plot([sys, sys], [rts[:,region].min(), rts[:,region].max()], c='r')
		# 	plt.legend()
		# 	plt.title(f"rts_{region}")
		# 	plt.savefig(self.output()[f"rts_{region}"].path)
		#
		# plt.figure(figsize=(15,7))
		# plt.plot(gts[:], label='strain_dl')
		# for i in range(v["dyastole list"].shape[0]):
		# 	dyas = v["dyastole list"][i]
		# 	plt.plot([dyas, dyas], [gts.min(), gts.max()], c='b')
		# for i in range(v["systole list"].shape[0]):
		# 	sys = v["systole list"][i]
		# 	plt.plot([sys, sys], [gts.min(), gts.max()], c='r')
		# plt.title("gts")
		# plt.legend()
		# plt.savefig(self.output()["gts"].path)

		f.close()
		g.close()
		h.close()
		v.close()
		m.close()
		os.rename(h5_path, self.output()['h5'].path)

	def output(self):
		try:
			os.makedirs(os.path.join(report_folder, self.dcm_name,self.__class__.__name__))
		except:
			pass
		dic = {}
		# dic['gls'] = luigi.LocalTarget(os.path.join(report_folder, self.dcm_name,self.__class__.__name__, f"gls.png"))
		# dic['gts'] = luigi.LocalTarget(os.path.join(report_folder, self.dcm_name,self.__class__.__name__, f"gts.png"))
		# for i in range(6):
		# 	dic[f'rls_{i}'] = luigi.LocalTarget(os.path.join(report_folder, self.dcm_name,self.__class__.__name__, f"rls_{i}.png"))
		# 	dic[f'rts_{i}'] = luigi.LocalTarget(os.path.join(report_folder, self.dcm_name,self.__class__.__name__, f"rts_{i}.png"))
		dic['h5'] = luigi.LocalTarget(os.path.join(report_folder, self.dcm_name,self.__class__.__name__, f"strain.h5"))
		return dic

class TrackDCM(luigi.Task):
	resources = {"gpu": 1}
	dcm_name = luigi.Parameter()

	window_halfsize = luigi.IntParameter(default = 6)
	region_halfsize = luigi.IntParameter(default = 4)

	iterations = luigi.IntParameter(default = 10)
	penalty_coefficient = luigi.IntParameter(default = 1800)
	smoothing_halfsize = luigi.IntParameter(default = 22)

	def requires(self):
		return SumSquaredDifferencesDCM(dcm_name = self.dcm_name, window_halfsize = self.window_halfsize, region_halfsize = self.region_halfsize)

	def run(self):
		setupLog(self.__class__.__name__)
		f = h5py.File(self.input().path, "r")
		h5_path = os.path.join(tmp_folder,self.__class__.__name__,self.dcm_name+'_tracking.h5')
		try:
			os.makedirs(os.path.join(tmp_folder,self.__class__.__name__))
		except:
			pass
		g = h5py.File(h5_path, "w")

		shape = f["ssd"].shape
		g.create_dataset("tracking", (shape[0], shape[1],shape[2],2), chunks = (4, shape[1], shape[2],2), dtype = "float32")


		for frame in range(shape[0]):
			ssd = f["ssd"][frame,...]
			g['tracking'][frame,...] = img_tracking_subpixel(ssd)

		for j in range(self.iterations):
			for frame in range(shape[0]):
				if(frame == 0):
					avg_track = (g['tracking'][frame,...]+g['tracking'][frame+1,...])/2.
				elif(frame==shape[0]-1):
					avg_track = (g['tracking'][frame-1,...]+g['tracking'][frame,...])/2.
				else:
					avg_track = (g['tracking'][frame-1,...]+g['tracking'][frame,...]+g['tracking'][frame+1,...])/3.
				blurred_track = block_blur(avg_track, self.window_halfsize, smoothing_halfsize=self.smoothing_halfsize)

				ssd = f["ssd"][frame,...]
				ssd2 = ssd+self.penalty_coefficient*penalty_term(blurred_track, self.window_halfsize)
				g['tracking'][frame,...] = img_tracking_subpixel(ssd2)

		for frame in range(shape[0]):
			if(frame == 0):
				avg_track = (g['tracking'][frame,...]+g['tracking'][frame+1,...])/2.
			elif(frame==shape[0]-1):
				avg_track = (g['tracking'][frame-1,...]+g['tracking'][frame,...])/2.
			else:
				avg_track = (g['tracking'][frame-1,...]+g['tracking'][frame,...]+g['tracking'][frame+1,...])/3.
			g['tracking'][frame,...] = block_blur(avg_track, self.window_halfsize, smoothing_halfsize=self.smoothing_halfsize)

		f.close()
		g.close()
		os.rename(h5_path, self.output().path)
		os.remove(self.input().path)

	def output(self):
		try:
			os.makedirs(os.path.join(report_folder, self.dcm_name,self.__class__.__name__))
		except:
			pass
		return luigi.LocalTarget(os.path.join(report_folder, self.dcm_name,self.__class__.__name__, f"tracking_w{self.window_halfsize}_r{self.region_halfsize}_i{self.iterations}_p{self.penalty_coefficient}_s{self.smoothing_halfsize}.h5"))


class TrackDCM_backup(luigi.Task):
	dcm_name = luigi.Parameter()

	window_halfsize = luigi.IntParameter(default = 6)
	region_halfsize = luigi.IntParameter(default = 4)

	iterations = luigi.IntParameter(default = 10)
	penalty_coefficient = luigi.IntParameter(default = 180)
	smoothing_halfsize = luigi.IntParameter(default = 22)

	def requires(self):
		return SumSquaredDifferencesDCM(dcm_name = self.dcm_name, window_halfsize = self.window_halfsize, region_halfsize = self.region_halfsize)

	def run(self):
		setupLog(self.__class__.__name__)
		f = h5py.File(self.input().path, "r")
		h5_path = os.path.join(tmp_folder,self.__class__.__name__,self.dcm_name+'_tracking.h5')
		try:
			os.makedirs(os.path.join(tmp_folder,self.__class__.__name__))
		except:
			pass
		g = h5py.File(h5_path, "w")

		shape = f["ssd"].shape
		g.create_dataset("tracking", (shape[0], shape[1],shape[2],2), chunks = (4, shape[1], shape[2],2), dtype = "float32")

		for frame in range(shape[0]):
			ssd = f["ssd"][frame,...]
			track = img_tracking_subpixel(ssd)

			for j in range(self.iterations):
				blurred_track = block_blur(track, self.window_halfsize, smoothing_halfsize=self.smoothing_halfsize)
				ssd += self.penalty_coefficient*penalty_term(blurred_track, self.window_halfsize)
				track = img_tracking_subpixel(ssd)

			g['tracking'][frame,...] = block_blur(track, self.window_halfsize, smoothing_halfsize=self.smoothing_halfsize) - self.window_halfsize

		f.close(9)
		g.close()
		os.rename(h5_path, self.output().path)

	def output(self):
		try:
			os.makedirs(os.path.join(report_folder, self.dcm_name,self.__class__.__name__))
		except:
			pass
		return luigi.LocalTarget(os.path.join(report_folder, self.dcm_name,self.__class__.__name__, f"tracking_w{self.window_halfsize}_r{self.region_halfsize}_i{self.iterations}_p{self.penalty_coefficient}_s{self.smoothing_halfsize}.h5"))

# class SumSquaredDifferencesDCM(luigi.Task):
# 	dcm_name = luigi.Parameter()
# 	window_halfsize = luigi.IntParameter(default = 6)
# 	region_halfsize = luigi.IntParameter(default = 4)
#
# 	def run(self):
# 		setupLog(self.__class__.__name__)
# 		file_path = os.path.join(dicom_folder,self.dcm_name+'.dcm')
# 		img = read_dicom(file_path)
#
# 		h5_path = os.path.join(tmp_folder,self.__class__.__name__,self.dcm_name+'_ssd.h5')
# 		try:
# 			os.makedirs(os.path.join(tmp_folder,self.__class__.__name__))
# 		except:
# 			pass
# 		f = h5py.File(h5_path, "w")
#
# 		f.create_dataset("ssd", (img.shape[0]-1, img.shape[1], img.shape[2], 2*self.window_halfsize+1, 2*self.window_halfsize+1),
# 						chunks = (4, img.shape[1], img.shape[2], 2*self.window_halfsize+1, 2*self.window_halfsize+1), dtype = "float32")
#
# 		for frame in range(img.shape[0]-1):
# 			f["ssd"][frame,...] = squared_image_differences(img[frame+1,...,0], img[frame,...,0], region_halfsize=self.region_halfsize, window_halfsize = self.window_halfsize)
#
# 		f.close()
# 		os.rename(h5_path, self.output().path)
#
# 	def output(self):
# 		try:
# 			os.makedirs(os.path.join(report_folder, self.dcm_name,self.__class__.__name__))
# 		except:
# 			pass
# 		return luigi.LocalTarget(os.path.join(report_folder, self.dcm_name,self.__class__.__name__, f"sumsquareddifferences_w{self.window_halfsize}_r{self.region_halfsize}.h5"))

class SumSquaredDifferencesDCM(luigi.Task):
	dcm_name = luigi.Parameter()
	window_halfsize = luigi.IntParameter(default = 6)
	region_halfsize = luigi.IntParameter(default = 4)

	def requires(self):
		file_path = os.path.join(dicom_folder,self.dcm_name+'.dcm')
		img = read_dicom(file_path)
		for frame in range(img.shape[0]-1):
			yield SumSquaredDifferencesFrameDCM(dcm_name = self.dcm_name, window_halfsize = self.window_halfsize, region_halfsize = self.region_halfsize, frame = frame)

	def run(self):
		setupLog(self.__class__.__name__)
		file_path = os.path.join(dicom_folder,self.dcm_name+'.dcm')
		img = read_dicom(file_path)

		h5_path = os.path.join(tmp_folder,self.__class__.__name__,self.dcm_name+'_ssd.h5')
		try:
			os.makedirs(os.path.join(tmp_folder,self.__class__.__name__))
		except:
			pass
		f = h5py.File(h5_path, "w")

		first = True
		for task in self.requires():
			file_path = task.output().path
			g = h5py.File(file_path, 'r')
			if first:
				f.create_dataset("ssd", (img.shape[0]-1, g["ssd"].shape[0], g["ssd"].shape[1], 2*self.window_halfsize+1, 2*self.window_halfsize+1),
								chunks = (4, g["ssd"].shape[0], g["ssd"].shape[1], 2*self.window_halfsize+1, 2*self.window_halfsize+1),
								dtype = "float32")
				first = False
			f["ssd"][task.frame,...] = g["ssd"][...]
			g.close()

		f.close()
		os.rename(h5_path, self.output().path)
		for task in self.requires():
			os.remove(task.output().path)

	def output(self):
		try:
			os.makedirs(os.path.join(report_folder, self.dcm_name,self.__class__.__name__))
		except:
			pass
		return luigi.LocalTarget(os.path.join(report_folder, self.dcm_name,self.__class__.__name__, f"sumsquareddifferences_w{self.window_halfsize}_r{self.region_halfsize}.h5"))

class SumSquaredDifferencesFrameDCM(luigi.Task):
	resources = {'cpu': 1}
	dcm_name = luigi.Parameter()
	window_halfsize = luigi.IntParameter(default = 6)
	region_halfsize = luigi.IntParameter(default = 4)
	frame = luigi.IntParameter()

	def requires(self):
		return {"Segment Ventricle":SegmentVentricleDCM(dcm_name = self.dcm_name),
				"Info":InfoDCM(dcm_name = self.dcm_name)}

	def run(self):
		setupLog(self.__class__.__name__)

		with open(self.input()['Info']['pickle'].path, 'rb') as handle:
			info = pickle.load(handle)

		file_path = os.path.join(dicom_folder,self.dcm_name+'.dcm')
		img = read_dicom(file_path)

		if (info["min x0"] is not None):
			img = img[:,info["min y0"]:info["max y1"], info["min x0"]:info["max x1"],...]

		h5_path = os.path.join(tmp_folder,self.__class__.__name__,self.dcm_name,f'{self.frame}_ssd.h5')
		try:
			os.makedirs(os.path.join(tmp_folder,self.__class__.__name__,self.dcm_name))
		except:
			pass
		f = h5py.File(h5_path, "w")
		g = h5py.File(self.input()["Segment Ventricle"].path, "r")

		img = img[:,g["bounding box top left"][0]:g["bounding box bottom right"][0],
					g["bounding box top left"][1]:g["bounding box bottom right"][1],...]

		f.create_dataset("ssd", (img.shape[1], img.shape[2], 2*self.window_halfsize+1, 2*self.window_halfsize+1),
						chunks = (img.shape[1], img.shape[2], 2*self.window_halfsize+1, 2*self.window_halfsize+1),
						dtype = "float32")
		f["ssd"][...] = squared_image_differences_fwd(img[self.frame+1,...,0], img[self.frame,...,0], region_halfsize=self.region_halfsize, window_halfsize = self.window_halfsize)

		f.close()
		g.close()
		os.rename(h5_path, self.output().path)

	def output(self):
		try:
			os.makedirs(os.path.join(report_folder, self.dcm_name,self.__class__.__name__))
		except:
			pass
		return luigi.LocalTarget(os.path.join(report_folder, self.dcm_name,self.__class__.__name__, f"sumsquareddifferences_w{self.window_halfsize}_r{self.region_halfsize}_f{self.frame}.h5"))

class MyocardiumRegionsDCM(luigi.Task):
	dcm_name = luigi.Parameter()

	def requires(self):
		return {'segmentation': SegmentVentricleDCM(dcm_name = self.dcm_name),
				'segpoints': SegmentPointsDCM(dcm_name = self.dcm_name),
				'volumes': VolumeCurvesDCM(dcm_name = self.dcm_name)}

	def run(self):
		setupLog(self.__class__.__name__)

		h5_path = os.path.join(tmp_folder,self.__class__.__name__,self.dcm_name+'_myo_regions_segmentation.h5')
		try:
			os.makedirs(os.path.join(tmp_folder,self.__class__.__name__))
		except:
			pass
		f = h5py.File(h5_path, "w")
		g = h5py.File(self.input()['segmentation'].path, "r")
		h = h5py.File(self.input()['segpoints'].path, "r")
		v = h5py.File(self.input()['volumes']['h5'].path, "r")

		vol_curve = v["volume curve smoothed"][...]
		teledyastole = vol_curve.argmax()
		td_mask = (g["segmentation avg"][teledyastole,...,0]==1)
		xx = np.tensordot(np.arange(td_mask.shape[0]), np.ones(td_mask.shape[1]), axes=0)
		yy = np.tensordot(np.ones(td_mask.shape[0]), np.arange(td_mask.shape[1]), axes=0)

		n=td_mask.sum()
		std_r = np.sqrt((xx**2*td_mask/n).sum()-(xx*td_mask/n).sum()**2+(yy**2*td_mask/n).sum()-(yy*td_mask/n).sum()**2)
		myo_thickness = int(std_r/2)+1


		# All related to the average segmentation, it should be possible to do the same with avg+1, avg-1 and original
		f.create_dataset(f"avg myo seg", (*(g["segmentation original"].shape[:3]),1),
						chunks = (4, *(g["segmentation original"].shape[1:3]),1), dtype = "int32",
						compression='gzip', compression_opts=9)

		f.create_dataset(f"avg seg myo regions", (*(g["segmentation original"].shape[:3]), 6),
						chunks = (4, *(g["segmentation original"].shape[1:3]), 1), dtype = "int32",
						compression='gzip', compression_opts=9)

		echo_segmentation = (g[f"segmentation avg"][...]==1).astype(int)

		for frame in range(echo_segmentation.shape[0]):
			[x_ap,y_ap] = h["avg seg points"][frame,0,:]
			[x_mpc,y_mpc] = h["avg seg points"][frame,1,:]
			[x_mp1,y_mp1] = h["avg seg points"][frame,2,:]
			[x_mp2,y_mp2] = h["avg seg points"][frame,3,:]

			echo_segmentation[frame,...,0] = estimate_myocardium_1mask(echo_segmentation[frame,...,0],x_mp1, y_mp1, x_mp2, y_mp2, myo_thickness=myo_thickness)
			region_list = find_myo_regions(echo_segmentation[frame,...,0], x_mpc, y_mpc, x_ap, y_ap)
			for region in range(6):
				f[f"avg seg myo regions"][frame,...,region] = region_list[region]

		f[f"avg myo seg"][...] = echo_segmentation

		f.close()
		os.rename(h5_path, self.output().path)


	def output(self):
		try:
			os.makedirs(os.path.join(report_folder, self.dcm_name,self.__class__.__name__))
		except:
			pass
		return luigi.LocalTarget(os.path.join(report_folder, self.dcm_name,self.__class__.__name__, "myo_regions_segmentation.h5"))


class SegmentPointsDCM(luigi.Task):
	dcm_name = luigi.Parameter()
	resources = {"cpu":1}

	def requires(self):
		return SegmentVentricleDCM(dcm_name = self.dcm_name)

	def run(self):
		setupLog(self.__class__.__name__)
		# file_path = os.path.join(dicom_folder,self.dcm_name+'.dcm')
		# img = read_dicom(file_path)

		h5_path = os.path.join(tmp_folder,self.__class__.__name__,self.dcm_name+'_lv_points_segmentation.h5')
		try:
			os.makedirs(os.path.join(tmp_folder,self.__class__.__name__))
		except:
			pass
		f = h5py.File(h5_path, "w")
		g = h5py.File(self.input().path, "r")

		for modifier in ["original", "avg", "avg+1", "avg-1"]:
		# 0: apex, 1: mitral plane center, 2: mitral plane left edge, 3: mitral plane right edge
			f.create_dataset(f"{modifier} seg points", (g["segmentation original"].shape[0],4, 2), dtype = "float32")
			f.create_dataset(f"{modifier} seg disks", (g["segmentation original"].shape[0], 20, 2, 2), dtype = "float32")
			f.create_dataset(f"{modifier} seg disk thickness", (g["segmentation original"].shape[0]), dtype = "float32")

			# if modifier == "avg":
			# 	f.create_dataset(f"{modifier} seg myo regions", (*(g["segmentation original"].shape[:3]), 6),
			# 					chunks = (4, *(g["segmentation original"].shape[1:3]), 1), dtype = "int32",
			# 					compression='gzip', compression_opts=9)

			echo_segmentation = (g[f"segmentation {modifier}"][...]==1).astype(int)

			for frame in range(echo_segmentation.shape[0]):
				lv_mask = (echo_segmentation[frame,...,0]==1)
				if(lv_mask.sum()>0):
					(x_mpc,y_mpc), (x_mp1, y_mp1), (x_mp2, y_mp2), (x_ap, y_ap) = find_mitral_plane_and_apex_1mask(lv_mask)
					# echo_segmentation[frame,...,0] = estimate_myocardium_1mask(echo_segmentation[frame,...,0],x_mp1, y_mp1,x_mp2, y_mp2)

					coord_list = find_disks_1mask(lv_mask, x_mpc, y_mpc, x_ap, y_ap)

					f[f"{modifier} seg points"][frame,0,:] = [x_ap,y_ap]
					f[f"{modifier} seg points"][frame,1,:] = [x_mpc,y_mpc]
					f[f"{modifier} seg points"][frame,2,:] = [x_mp1,y_mp1]
					f[f"{modifier} seg points"][frame,3,:] = [x_mp2,y_mp2]
					f[f"{modifier} seg disk thickness"][frame] = np.sqrt((x_ap-x_mpc)**2+(y_ap-y_mpc)**2)/20.
					for disk in range(len(coord_list)):
						f[f"{modifier} seg disks"][frame, disk,0,:] = coord_list[disk][0:2]
						f[f"{modifier} seg disks"][frame, disk,1,:] = coord_list[disk][2:4]
			# 		if modifier == "avg":
			# 			region_list = find_myo_regions(echo_segmentation[frame,...,0], x_mpc, y_mpc, x_ap, y_ap)
			# 			for region in range(6):
			# 				f[f"{modifier} seg myo regions"][frame,...,region] = region_list[region]
			#
			# f[f"{modifier} seg"][...] = echo_segmentation

		f.close()
		g.close()
		os.rename(h5_path, self.output().path)

	def output(self):
		try:
			os.makedirs(os.path.join(report_folder, self.dcm_name,self.__class__.__name__))
		except:
			pass
		return luigi.LocalTarget(os.path.join(report_folder, self.dcm_name,self.__class__.__name__, "lv_points_segmentation.h5"))

class SegmentVentricleDCM(luigi.Task):
	resources = {"gpu": 1}
	dcm_name = luigi.Parameter()
	n_augmentations = luigi.IntParameter(default = 20)

	def requires(self):
		return 	{'Info':InfoDCM(dcm_name=self.dcm_name)}

	def run(self):
		setupLog(self.__class__.__name__)

		with open(self.input()['Info']['pickle'].path, 'rb') as handle:
			info = pickle.load(handle)

		file_path = os.path.join(dicom_folder,self.dcm_name+'.dcm')
		img = read_dicom(file_path)

		if (info["min x0"] is not None):
			img = img[:,info["min y0"]:info["max y1"], info["min x0"]:info["max x1"],...]

		h5_path = os.path.join(tmp_folder,self.__class__.__name__,self.dcm_name+'_segmentation.h5')
		try:
			os.makedirs(os.path.join(tmp_folder,self.__class__.__name__))
		except:
			pass
		f = h5py.File(h5_path, "w")

		f.create_dataset("segmentation original", (*(img.shape[:3]), 1),
						chunks = (4, *(img.shape[1:3]), 1), dtype = "int32", compression='gzip', compression_opts=9)

		f.create_dataset("augmented segmentations", (self.n_augmentations, *(img.shape[:3]), 1),
						chunks = (1, 4, *(img.shape[1:3]), 1), dtype = "int32", compression='gzip', compression_opts=9)

		f.create_dataset("segmentation avg", (*(img.shape[:3]), 1),
						chunks = (4, *(img.shape[1:3]), 1), dtype = "int32", compression='gzip', compression_opts=9)
		f.create_dataset("segmentation avg+1", (*(img.shape[:3]), 1),
						chunks = (4, *(img.shape[1:3]), 1), dtype = "int32", compression='gzip', compression_opts=9)
		f.create_dataset("segmentation avg-1", (*(img.shape[:3]), 1),
						chunks = (4, *(img.shape[1:3]), 1), dtype = "int32", compression='gzip', compression_opts=9)

		f.create_dataset("bounding box top left", (2), dtype = "int32")
		f.create_dataset("bounding box bottom right", (2), dtype = "int32")
		f.create_dataset("bounding box shape", (2), dtype = "int32")

		nn = UserSegmenter()
		f["segmentation original"][...,0] = nn.segment(img)

		for i in range(self.n_augmentations):
			img_augmented = random_augmentation(img, seed = i)
			augmented_segmentation = nn.segment(img_augmented)
			restored_segmentation = restore_augmentation(augmented_segmentation, seed = i)
			f["augmented segmentations"][i,...,0] = restored_segmentation

		f["segmentation avg"][...] = ((f["augmented segmentations"][...]==1).sum(axis=0)>(self.n_augmentations/2)).astype(int)
		f["segmentation avg"][...,0] = postprocess_1mask(f["segmentation avg"][...,0])

		f["segmentation avg+1"][...] = ((f["augmented segmentations"][...]==1).sum(axis=0)>(self.n_augmentations/2*1.68)).astype(int)
		f["segmentation avg+1"][...,0] = postprocess_1mask(f["segmentation avg+1"][...,0])

		f["segmentation avg-1"][...] = ((f["augmented segmentations"][...]==1).sum(axis=0)>(self.n_augmentations/2*0.32)).astype(int)
		f["segmentation avg-1"][...,0] = postprocess_1mask(f["segmentation avg-1"][...,0])

		margin=100
		roi = (f["segmentation avg"][...]==1)
		f["bounding box top left"][0] = max(0,roi.max(axis=0).max(axis=1).argmax()-margin)
		f["bounding box top left"][1] = max(0,roi.max(axis=0).max(axis=0).argmax()-margin)
		f["bounding box bottom right"][0] = min(img.shape[1],img.shape[1]-roi.max(axis=0).max(axis=1)[::-1].argmax()+margin)
		f["bounding box bottom right"][1] = min(img.shape[2],img.shape[2]-roi.max(axis=0).max(axis=0)[::-1].argmax()+margin)
		f["bounding box shape"][0] = f["bounding box bottom right"][0] - f["bounding box top left"][0]
		f["bounding box shape"][1] = f["bounding box bottom right"][1] - f["bounding box top left"][1]

		f.close()
		os.rename(h5_path, self.output().path)

	def output(self):
		try:
			os.makedirs(os.path.join(report_folder, self.dcm_name,self.__class__.__name__))
		except:
			pass
		return luigi.LocalTarget(os.path.join(report_folder, self.dcm_name,self.__class__.__name__, "segmentation.h5"))


class ProcessStrainDCM(luigi.Task):
	dcm_name = luigi.Parameter()

	def requires(self):
		return {'VideoSegmentation':VideoSegmentationDCM(dcm_name=self.dcm_name, myo_segmentation=True),
				'VideoTracking':VideoTrackingDCM(dcm_name=self.dcm_name),
				'StrainCurves':StrainCurves(dcm_name=self.dcm_name),
				'MeasurementsStrain': MeasurementsStrainDCM(dcm_name=self.dcm_name)}

	def run(self):
		setupLog(self.__class__.__name__)
		with open(self.output().path,'w') as f:
			f.write(f"ProcessDCM {self.dcm_name} finished\n")

	def output(self):
		TIMESTRING=dt.datetime.now().strftime("%y%m%d-%H%M%S")
		return luigi.LocalTarget(os.path.join(log_path, f"Process_{self.dcm_name}-{TIMESTRING}.txt"))

class ProcessVolumesDCM(luigi.Task):
	dcm_name = luigi.Parameter()

	def requires(self):
		return {'VideoSegmentation':VideoSegmentationDCM(dcm_name=self.dcm_name, myo_segmentation=True),
				'VolumeCurves':VolumeCurvesDCM(dcm_name=self.dcm_name),
				'VolumeGraphs':VolumeGraphs(dcm_name=self.dcm_name),
				'Info':InfoDCM(dcm_name=self.dcm_name),
				'MeasurementsVolumes':MeasurementsVolumesDCM(dcm_name=self.dcm_name),
				'Confidence':ConfidenceCheck(dcm_name=self.dcm_name),
				'MeasurementsGT': MeasurementsGT(dcm_name=self.dcm_name),
				'GTMasksAndVolumes': GTMasksAndVolumes(dcm_name=self.dcm_name),}

	def run(self):
		setupLog(self.__class__.__name__)
		with open(self.output().path,'w') as f:
			f.write(f"ProcessDCM {self.dcm_name} finished\n")

	def output(self):
		TIMESTRING=dt.datetime.now().strftime("%y%m%d-%H%M%S")
		return luigi.LocalTarget(os.path.join(log_path, f"Process_{self.dcm_name}-{TIMESTRING}.txt"))

class ProcessGTSpeckle(luigi.Task):
	dcm_name = luigi.Parameter()

	def requires(self):
		return {'VideoTrackingGT':VideoTrackingGT(dcm_name=self.dcm_name)}
				# 'VideoTrackingGT':VideoTrackingDCM(dcm_name=self.dcm_name),

	def run(self):
		setupLog(self.__class__.__name__)
		with open(self.output().path,'w') as f:
			f.write(f"ProcessDCM {self.dcm_name} finished\n")

	def output(self):
		TIMESTRING=dt.datetime.now().strftime("%y%m%d-%H%M%S")
		return luigi.LocalTarget(os.path.join(log_path, f"Process_{self.dcm_name}-{TIMESTRING}.txt"))

class AllVolumes(luigi.WrapperTask):
	def requires(self):
		for filename in os.listdir(dicom_folder):
			if (len(filename)>3):
				if(filename[-3:]=='dcm'):
					yield ProcessVolumesDCM(dcm_name = filename[:-4])
				if(filename[-3:]=='gtp'):
					yield ProcessGTSpeckle(dcm_name = filename[:-4])
	def run(self):
		setupLog(self.__class__.__name__)
		with open(self.output().path,'w') as f:
			f.write(f"AllVolumes finished\n")

	def output(self):
		TIMESTRING=dt.datetime.now().strftime("%y%m%d-%H%M%S")
		return luigi.LocalTarget(os.path.join(log_path, f"AllVolumes-{TIMESTRING}.txt"))

class AllTasks(luigi.WrapperTask):
	def requires(self):
		for filename in os.listdir(dicom_folder):
			if (len(filename)>3):
				if(filename[-3:]=='dcm'):
					yield ProcessVolumesDCM(dcm_name = filename[:-4])
					yield ProcessStrainDCM(dcm_name = filename[:-4])
				if(filename[-3:]=='gtp'):
					yield ProcessGTSpeckle(dcm_name = filename[:-4])
