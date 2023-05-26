import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from scipy.ndimage.interpolation import rotate
from PIL import Image
import cv2 as cv

import json
from detectron2.data import DatasetCatalog, MetadataCatalog

def calc_iou(x, y):
	return np.bitwise_and(x, y).sum() / np.bitwise_or(x, y).sum() 

def update_dict(old_dict, new_dict):
	for key in new_dict.keys():
		try:
			old_dict[key].extend(new_dict[key])
		except:
			old_dict[key] = new_dict[key]
	return old_dict

def match_transforms(indxsA, indxsB, boxA, boxB, outputA, outputB):

	indxs_match = {}
	skip = []
	for idxa in indxsA[::-1]:
		for idxb in indxsB[::-1]:
			if idxb in skip: continue
			iou = calc_iou(boxA[idxa], boxB[idxb])
			if iou > 0.7:
				indxs_match[idxa] = [[max(outputA.scores[idxa].item(), outputB.scores[idxb].item()), max(outputA.pred_masks[idxa].sum().item(), outputB.pred_masks[idxb].sum().item())]]
				skip.append(idxb)
				break

	return indxs_match

def do_match(model, file_name):
	temp_dicts = {'classes': [], 'scores': [], 'boxes': []}

	img_orig = cv.imread(file_name)
	H, W, _ = img_orig.shape
	
	if W*H < 5000:
		temp_dicts['classes'].append([])
		temp_dicts['scores'].append([])
		temp_dicts['boxes'].append([])
		return temp_dicts 
	
	img_vflp = np.flip(img_orig, 0)
	img_hflp = np.flip(img_orig, 1)
	img_rot = rotate(img_orig, 90)

	output_orig = model(img_orig)['instances']
	output_vflp = model(img_vflp)['instances']
	output_hflp = model(img_hflp)['instances']
	output_rot = model(img_rot)['instances']

	bboxes_orig = [box.detach().cpu().numpy().astype(int) for box in output_orig.pred_boxes]
	drawn_orig = [cv.rectangle(np.zeros((H,W)), box[:2], box[2:], 1, -1).astype(bool) for box in bboxes_orig]
	bboxes_vflp = [box.detach().cpu().numpy().astype(int) for box in output_vflp.pred_boxes]
	drawn_vflp = [np.flip(cv.rectangle(np.zeros((H,W)), box[:2], box[2:], 1, -1), 0).astype(bool) for box in bboxes_vflp]
	bboxes_hflp = [box.detach().cpu().numpy().astype(int) for box in output_hflp.pred_boxes]
	drawn_hflp = [np.flip(cv.rectangle(np.zeros((H,W)), box[:2], box[2:], 1, -1), 1).astype(bool) for box in bboxes_hflp]
	bboxes_rot = [box.detach().cpu().numpy().astype(int) for box in output_rot.pred_boxes]
	drawn_rot = [rotate(cv.rectangle(np.zeros((W,H)), box[:2], box[2:], 1, -1).astype(bool), -90) for box in bboxes_rot]

	indxs_orig = list(range(len(bboxes_orig)))
	indxs_vflp = list(range(len(bboxes_vflp)))
	indxs_hflp = list(range(len(bboxes_hflp)))
	indxs_rot = list(range(len(bboxes_rot)))

	# ov
	indxs_matched = match_transforms(indxs_orig.copy(), indxs_vflp.copy(), drawn_orig, drawn_vflp, output_orig, output_vflp)
	# oh
	update_dict(indxs_matched, match_transforms(indxs_orig.copy(), indxs_hflp.copy(), drawn_orig, drawn_hflp, output_orig, output_hflp))
	# or
	update_dict(indxs_matched, match_transforms(indxs_orig.copy(), indxs_rot.copy(), drawn_orig, drawn_rot, output_orig, output_rot))
 
	for key, value in indxs_matched.items():
		if len(value) >= 2:
			temp_dicts['classes'].append(output_orig.pred_classes[key].item())
			temp_dicts['scores'].append(output_orig.scores[key].item())
			temp_dicts['boxes'].append(bboxes_orig[key])
	return temp_dicts

def run_tta(model, eval_image):
	results_dicts = do_match(model, eval_image)
	return results_dicts

def get_xray_dicts(mode, cfg):

	if mode == "det":
		with open(cfg.DATASETS.DET_TRAIN_JSON, "r") as sn:
			obj_sep_train = json.load(sn)
		return obj_sep_train
	elif mode == "segm":
		with open(cfg.DATASETS.SEGM_TRAIN_JSON, "r") as sn:
			obj_sep_train = json.load(sn)
		return obj_sep_train
	elif mode == "test":
		with open(cfg.DATASETS.TEST_JSON, "r") as sn:
			obj_sep_test = json.load(sn)
		return obj_sep_test
	else:
		assert False, "[{}] is not part of accepted modes: train, test.".format(mode)

def register_dataset(cfg):
	for d in ["det", "segm", "test"]:
		DatasetCatalog.register("xray_" + d, lambda d=d: get_xray_dicts(d, cfg))
		MetadataCatalog.get("xray_" + d).set(thing_classes=["Gun", "Knife", "Wrench", "Pliers", "Scissors"])