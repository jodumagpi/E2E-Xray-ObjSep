import copy, os, logging
from collections import OrderedDict
from typing import Set, Any, Dict, List

import torch
from torch.nn.parallel import DistributedDataParallel
import torch.utils.data as torchdata

import detectron2.utils.comm as comm
import detectron2.data.transforms as T
from detectron2.solver.build import maybe_add_gradient_clipping, reduce_param_groups
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch, DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.solver import build_lr_scheduler
from detectron2.data import ( build_batch_data_loader, get_detection_dataset_dicts, 
                             DatasetMapper, MapDataset, DatasetFromList,
                             MetadataCatalog, build_detection_test_loader )
from detectron2.data.samplers import TrainingSampler, RepeatFactorTrainingSampler, RandomSubsetTrainingSampler
from detectron2.utils.events import EventStorage
from detectron2.utils.logger import _log_api_usage, setup_logger
from detectron2.config import global_cfg

from detectron2.engine import DefaultPredictor
import cv2 as cv
import numpy as np
from sklearn import metrics
from tqdm import tqdm
import pandas as pd
import json
import time
import cv2 as cv

from data import *

setup_logger()

logger = logging.getLogger("detectron2")

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def do_eval(cfg, eval_image, tta=None):
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
	cfg.TEST.DETECTIONS_PER_IMAGE = 10
	
	logger.info("Evaluating from weights: {}".format(cfg.MODEL.WEIGHTS))
	
	start = time.time()
	predictor = DefaultPredictor(cfg)
	logger.info("Loading the model took {} seconds".format(time.time()-start))
	
	start = time.time()
	if tta:
		logger.info("Running tests with TTA.")
		results_dict = run_tta(predictor, eval_image)
	else:
		logger.info("Running tests without TTA.")
		input = cv.imread(eval_image)
		H, W, _ = input.shape
		preds = predictor(input)['instances']
		results_dict = {
				'classes' : preds.pred_classes.cpu().numpy().tolist(),
				'scores' : preds.scores.cpu().numpy().tolist(),
				'boxes' : [box.detach().cpu().numpy().astype(int) for box in preds.pred_boxes]
				}

	logger.info("Prediciton took {} seconds".format(time.time()-start))
	
	print(results_dict)
	
	return results_dict

def do_test(cfg, model, mask=True):
	results = OrderedDict()
	loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST)
	evaluator = COCOEvaluator("xray_test", tasks=["segm" if mask else "bbox"], use_fast_impl=True,
										output_dir=os.path.join(cfg.OUTPUT_DIR, "eval"))
	results = inference_on_dataset(model, loader, evaluator)["segm" if mask else "bbox"]
	return results['AP']
	
def build_train_loader(cfg, mask=True):
	dataset = get_detection_dataset_dicts(
						cfg.DATASETS.TRAIN[1] if mask else cfg.DATASETS.TRAIN[0],
						filter_empty=False,
						min_keypoints=0,
						proposal_files=None,
					)
	
	_log_api_usage("dataset." + cfg.DATASETS.TRAIN[0])
	
	logger.info("Total number of images: {}".format(len(dataset)))
	
	mapper = DatasetMapper(cfg, is_train=True, 
	                       augmentations=[T.ResizeShortestEdge(short_edge_length=(640, 672, 704, 736, 768, 800), max_size=1333, sample_style='choice'), 
										T.RandomFlip(), # horizontal
										T.RandomFlip(horizontal=False, vertical=True)])
	dataset = MapDataset(dataset, mapper)
	
	loader_size = int(len(dataset)/cfg.SOLVER.IMS_PER_BATCH)
	
	sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
	logger.info("Using training sampler {}".format(sampler_name))
	if sampler_name == "TrainingSampler":
		sampler = TrainingSampler(len(dataset), shuffle=False) # let's not shuffle what's already shuffled
	elif sampler_name == "RepeatFactorTrainingSampler":
		repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
				dataset, cfg.DATALOADER.REPEAT_THRESHOLD
			)
		sampler = RepeatFactorTrainingSampler(repeat_factors)
	elif sampler_name == "RandomSubsetTrainingSampler":
		sampler = RandomSubsetTrainingSampler(len(dataset), cfg.DATALOADER.RANDOM_SUBSET_RATIO)
	else:
		raise ValueError("Unknown training sampler: {}".format(sampler_name))
	
	if isinstance(dataset, list):
		dataset = DatasetFromList(dataset, copy=False)
	if isinstance(dataset, torchdata.IterableDataset):
		assert sampler is None, "sampler must be None if dataset is IterableDataset"
	else:
		if sampler is None:
			sampler = TrainingSampler(len(dataset))
		assert isinstance(sampler, torchdata.Sampler), f"Expect a Sampler but got {type(sampler)}"
	
	return build_batch_data_loader(dataset,
									sampler,
									total_batch_size=cfg.SOLVER.IMS_PER_BATCH,
									aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
									num_workers= cfg.DATALOADER.NUM_WORKERS,)

def build_optimizer(cfg, model, mask=True):
	base_lr=cfg.SOLVER.BASE_LR
	weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM
	bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR
	weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS
	weight_decay = None
	overrides = None
	
	if overrides is None:
		overrides = {}
	defaults = {}
	if base_lr is not None:
		defaults["lr"] = base_lr
	if weight_decay is not None:
		defaults["weight_decay"] = weight_decay
	bias_overrides = {}
	if bias_lr_factor is not None and bias_lr_factor != 1.0:
		if base_lr is None:
			raise ValueError("bias_lr_factor requires base_lr")
		bias_overrides["lr"] = base_lr * bias_lr_factor
	if weight_decay_bias is not None:
		bias_overrides["weight_decay"] = weight_decay_bias
	if len(bias_overrides):
		if "bias" in overrides:
			raise ValueError("Conflicting overrides for 'bias'")
		overrides["bias"] = bias_overrides
	
	norm_module_types = (
		torch.nn.BatchNorm1d,
		torch.nn.BatchNorm2d,
		torch.nn.BatchNorm3d,
		torch.nn.SyncBatchNorm,
		torch.nn.GroupNorm,
		torch.nn.InstanceNorm1d,
		torch.nn.InstanceNorm2d,
		torch.nn.InstanceNorm3d,
		torch.nn.LayerNorm,
		torch.nn.LocalResponseNorm,
	)
	
	detection_only_modules = [module for module in model.backbone.modules()] + \
					[module for module in model.proposal_generator.modules()] + \
					[module for module in model.roi_heads.box_pooler.modules()] + \
					[module for module in model.roi_heads.box_head.modules()] + \
					[module for module in model.roi_heads.box_predictor.modules()]

	segmentation_only_modules = [module for module in model.roi_heads.mask_pooler.modules()] + \
                                [module for module in model.roi_heads.mask_head.modules()]
	
	params: List[Dict[str, Any]] = []
	memo: Set[torch.nn.parameter.Parameter] = set()
	
	modules = segmentation_only_modules if mask else detection_only_modules
	for module in modules:
		for module_param_name, value in module.named_parameters(recurse=False):
			if not value.requires_grad:
				continue
			if value in memo:
				continue
			memo.add(value)
	
			hyperparams = copy.copy(defaults)
			if isinstance(module, norm_module_types) and weight_decay_norm is not None:
				hyperparams["weight_decay"] = weight_decay_norm
			hyperparams.update(overrides.get(module_param_name, {}))
			params.append({"params": [value], **hyperparams})
	
	logger.info("Total number of params: {}".format(len(params))) 
	params = reduce_param_groups(params)
	logger.info("Total number of params (after reduction): {}".format(len(params))) 
	
	return maybe_add_gradient_clipping(cfg, torch.optim.SGD)(
		params,
		lr=cfg.SOLVER.BASE_LR,
		momentum=cfg.SOLVER.MOMENTUM,
		nesterov=cfg.SOLVER.NESTEROV,
		weight_decay=cfg.SOLVER.WEIGHT_DECAY,
	)

def do_train(cfg, model, resume=False):

	model.train()
	max_iter = cfg.SOLVER.MAX_ITER
	switch_iter = cfg.SOLVER.SWITCH_ITER
	pretrain_iter = cfg.SOLVER.PRETRAIN_ITER
	det_iter = max_iter/2
	segm_iter = max_iter/2
	base_lr = cfg.SOLVER.BASE_LR
	steps = cfg.SOLVER.STEPS
	assert pretrain_iter < det_iter, "Pretrain iteration should be less than the maximum iteration for the detection section."
	writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []
	
	if resume:
		det_iter -= cfg.SOLVER.DET_RESUME_ITER
		exponent = sum([1 for stp in steps if cfg.SOLVER.DET_RESUME_ITER/stp >= 1])
		cfg.SOLVER.BASE_LR = base_lr * (0.1)**exponent
		cfg.SOLVER.STEPS = tuple([stp-cfg.SOLVER.DET_RESUME_ITER for stp in steps[exponent:]]) if exponent < 3 else ()
		logger.info("Learning rate modified from {} to {}.".format(base_lr, cfg.SOLVER.BASE_LR))
		logger.info("Solver steps modified from {} to {}.".format(steps, cfg.SOLVER.STEPS))
	
	det_optim = build_optimizer(cfg, model, mask=False)
	det_sked = build_lr_scheduler(cfg, det_optim)
	if not os.path.exists(os.path.join(cfg.OUTPUT_DIR, "det")):
		os.mkdir(os.path.join(cfg.OUTPUT_DIR, "det"))
	det_checkpointer = DetectionCheckpointer(
		model, os.path.join(cfg.OUTPUT_DIR, "det"), optimizer=det_optim, scheduler=det_sked
	)
	det_periodic_checkpointer = PeriodicCheckpointer(
		det_checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
	)
	
	if resume:
		segm_iter -= cfg.SOLVER.SEGM_RESUME_ITER
		exponent = sum([1 for stp in steps if cfg.SOLVER.SEGM_RESUME_ITER/stp >= 1])
		cfg.SOLVER.BASE_LR = base_lr * (0.1)**exponent
		cfg.SOLVER.STEPS = tuple([stp-cfg.SOLVER.SEGM_RESUME_ITER for stp in steps[exponent:]]) if exponent < 3 else ()
		logger.info("Learning rate modified from {} to {}.".format(base_lr, cfg.SOLVER.BASE_LR))
		logger.info("Solver steps modified from {} to {}.".format(steps, cfg.SOLVER.STEPS))
	
	segm_optim = build_optimizer(cfg, model, mask=True)
	segm_sked = build_lr_scheduler(cfg, segm_optim)
	if not os.path.exists(os.path.join(cfg.OUTPUT_DIR, "segm")):
		os.mkdir(os.path.join(cfg.OUTPUT_DIR, "segm"))
	segm_checkpointer = DetectionCheckpointer(
		model, os.path.join(cfg.OUTPUT_DIR, "segm"), optimizer=segm_optim, scheduler=segm_sked
	)
	segm_periodic_checkpointer = PeriodicCheckpointer(
		segm_checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
	)
	
	segm_start_iter = (
		segm_checkpointer.resume_or_load(cfg.MODEL.SEGM_WEIGHTS, resume=resume).get("iteration", -1)
	)
	det_start_iter = (
		det_checkpointer.resume_or_load(cfg.MODEL.DET_WEIGHTS, resume=resume).get("iteration", -1)
	)

	if segm_start_iter < 0 or det_start_iter < 0:
		segm_start_iter = 0
		det_start_iter = 0
	
	assert segm_start_iter == cfg.SOLVER.SEGM_RESUME_ITER, "Segmentation iterations mismatch! {}, {}".format(segm_start_iter, cfg.SOLVER.SEGM_RESUME_ITER)
	assert det_start_iter == cfg.SOLVER.DET_RESUME_ITER, "Detection iterations mismatch! {}, {}".format(det_start_iter, cfg.SOLVER.DET_RESUME_ITER)
	
	best_bbox_AP = det_checkpointer.resume_or_load(cfg.MODEL.DET_WEIGHTS, resume=resume).get("AP")
	best_segm_AP = segm_checkpointer.resume_or_load(cfg.MODEL.SEGM_WEIGHTS, resume=resume).get("AP")

	if best_bbox_AP is None or best_segm_AP is None:
		best_bbox_AP = 0
		best_segm_AP = 0
	
	start_iter = (segm_start_iter + det_start_iter)
	
	det_dataloader = build_train_loader(cfg, False)
	segm_dataloader = build_train_loader(cfg, True)
		
	with EventStorage(start_iter) as storage:
		det_best_wts = {k: v for k, v in model.state_dict().items() if not k.startswith("roi_heads.mask")} 
		segm_best_wts = {k: v for k, v in model.state_dict().items() if k.startswith("roi_heads.mask")}
		prev_iter = start_iter
		logger.info("Training starts at iteration {}.".format(prev_iter))
		logger.info("Current best AP for [det] : {} and [segm] : {}.".format(best_bbox_AP, best_segm_AP))
		if pretrain_iter > 0:
			next_iter = pretrain_iter
			logger.info("Pretraining detection section for {} iterations.".format(pretrain_iter))
			global_cfg.GLOBAL.TRAIN_MASK = False
		else:
			next_iter = int((prev_iter + switch_iter)/switch_iter) * switch_iter
		while next_iter <= max_iter:
			if not global_cfg.GLOBAL.TRAIN_MASK and det_iter == 0:
				global_cfg.GLOBAL.TRAIN_MASK = not global_cfg.GLOBAL.TRAIN_MASK
				logger.info("Iterations for detection section is complete!")
			if global_cfg.GLOBAL.TRAIN_MASK and segm_iter == 0:
				global_cfg.GLOBAL.TRAIN_MASK = not global_cfg.GLOBAL.TRAIN_MASK
				logger.info("Iterations for object seperation section is complete!")
			logger.info("Training [{}] section from iteration {} to {}".format("segm" if global_cfg.GLOBAL.TRAIN_MASK else "det", prev_iter, next_iter))
			for data, iteration in zip(segm_dataloader if global_cfg.GLOBAL.TRAIN_MASK else det_dataloader, range(prev_iter, next_iter)):
				storage.iter = iteration
	
				loss_dict = model(data)
				losses = sum(loss_dict.values())
				assert torch.isfinite(losses).all(), loss_dict
	
				loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
				losses_reduced = sum(loss for loss in loss_dict_reduced.values())
				if comm.is_main_process():
					storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)
	
				if not global_cfg.GLOBAL.TRAIN_MASK: 
					det_optim.zero_grad()
					losses.backward()
					det_optim.step()
					storage.put_scalar("lr", det_optim.param_groups[0]["lr"], smoothing_hint=False)
					det_sked.step()
				else:
					segm_optim.zero_grad()
					losses.backward()
					segm_optim.step()
					storage.put_scalar("lr", segm_optim.param_groups[0]["lr"], smoothing_hint=False)
					segm_sked.step()
				
				if iteration - start_iter > 5 and (
					(iteration + 1) % global_cfg.GLOBAL.PRINT_FREQ == 0 or iteration == max_iter - 1
				):
					for writer in writers:
						writer.write()
				
				if ((iteration + 1) % cfg.SOLVER.CHECKPOINT_PERIOD == 0):
					#_ = os.system("fusermount -u /content/onedrive")
					#_ = os.system("nohup rclone --vfs-cache-max-age 720h --vfs-cache-mode writes mount onedrive: /content/onedrive &")
					pass
				
				det_periodic_checkpointer.step(iteration)
				segm_periodic_checkpointer.step(iteration)
				
				for mode in ["segm", "det"]:
					last_wts_path = os.path.join(cfg.OUTPUT_DIR, mode, "model_{:07d}.pth".format(iteration-cfg.SOLVER.CHECKPOINT_PERIOD))
					if os.path.exists(last_wts_path):
						os.remove(last_wts_path)
	
				if ((iteration + 1) % switch_iter == 0):
					now_AP = do_test(cfg, model, True if global_cfg.GLOBAL.TRAIN_MASK else False)
					if not global_cfg.GLOBAL.TRAIN_MASK:
						if now_AP > best_bbox_AP:
							best_bbox_AP = now_AP
							det_best_wts = {k: v for k, v in model.state_dict().items() if not k.startswith("roi_heads.mask")}
							torch.save({"model": det_best_wts, 
										"AP": now_AP,
                                        "iteration": iteration + 1},
										os.path.join(cfg.OUTPUT_DIR, "det_best_wts.pth"))
							logger.info("Best [det] weights are saved with AP = [{:0.4f}]".format(best_bbox_AP))
					else:
						if now_AP > best_segm_AP:
							best_segm_AP = now_AP
							segm_best_wts = {k: v for k, v in model.state_dict().items() if k.startswith("roi_heads.mask")}
							torch.save({"model": segm_best_wts, 
										"AP": now_AP,
                                        "iteration": iteration + 1},
										os.path.join(cfg.OUTPUT_DIR, "segm_best_wts.pth"))
							logger.info("Best [segm] weights are saved with AP = [{:0.4f}]".format(best_segm_AP))
					
					comm.synchronize()
				
				if cfg.SOLVER.LOAD_BEST:
					best_wts = det_best_wts
					best_wts.update(segm_best_wts)
					model.load_state_dict(best_wts)
				
				if not global_cfg.GLOBAL.TRAIN_MASK:
					det_iter -= 1
				else:
					segm_iter -= 1
			
				if pretrain_iter > 0 and iteration == pretrain_iter - 1:
					pretrain_iter = 0
					logger.info("Iterations for pretraining detection section is complete!")
					break
			
				if iteration == max_iter - 1:
					logger.info("Training all sections is complete!")
					break
			
			prev_iter = next_iter
			next_iter += switch_iter
			
			global_cfg.GLOBAL.TRAIN_MASK = not global_cfg.GLOBAL.TRAIN_MASK