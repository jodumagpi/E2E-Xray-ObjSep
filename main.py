from detectron2 import model_zoo
from detectron2.config import get_cfg, set_global_cfg
from detectron2.modeling import build_model

from data import register_dataset
from engine import do_train, do_eval
from models import ModROIHeads, DeepLabV3Plus

import argparse, torch

def setup(args):
	
	cfg = get_cfg()
	cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
	
	cfg.DATASETS.TRAIN = ("xray_det", "xray_segm")
	cfg.DATASETS.TEST = ("xray_test",)
	cfg.DATASETS.DET_TRAIN_JSON = args.det_train_json
	cfg.DATASETS.SEGM_TRAIN_JSON = args.segm_train_json
	cfg.DATASETS.TEST_JSON = args.test_json
	cfg.DATASETS.TEST_CSV = args.test_csv
	cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
	cfg.DATALOADER.NUM_WORKERS = 0
	
	cfg.SOLVER.IMS_PER_BATCH = 8
	cfg.SOLVER.BASE_LR = 0.0001
	cfg.SOLVER.WARMUP_ITERS = 0
	cfg.SOLVER.MAX_ITER = args.max_iter
	cfg.SOLVER.PRETRAIN_ITER = args.pretrain_iter
	cfg.SOLVER.SWITCH_ITER = args.switch_iter
	cfg.SOLVER.DET_RESUME_ITER = args.det_resume_iter
	cfg.SOLVER.SEGM_RESUME_ITER = args.segm_resume_iter
	cfg.SOLVER.STEPS = tuple(int(x*(args.max_iter/8)) for x in range(1, 4))
	cfg.SOLVER.CHECKPOINT_PERIOD = args.save_iter
	cfg.SOLVER.LOAD_BEST = args.load_best
	
	cfg.MODEL.META_ARCHITECTURE = "ModGeneralizedRCNN"
	cfg.MODEL.SEGM_WEIGHTS = args.segm_wts_path
	cfg.MODEL.DET_WEIGHTS = args.det_wts_path
	cfg.MODEL.WEIGHTS = args.wts_path
	cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
	cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
	cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES = True
	cfg.MODEL.ROI_HEADS.NAME = "ModROIHeads"
	cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 16
	cfg.MODEL.ROI_MASK_HEAD.NAME = "DeepLabV3Plus"
	
	cfg.OUTPUT_DIR = args.save_path
	cfg.GLOBAL.TRAIN_MASK = args.train_mask
	cfg.GLOBAL.PRINT_FREQ = args.print_freq
	cfg.GLOBAL.THRESHOLD = 0.5
	cfg.GLOBAL.HN_COUNT = [0] * 5
	cfg.GLOBAL.HP_COUNT = [0] * 5
	
	set_global_cfg(cfg)
	
	return cfg

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Train an end-to-end object separation network.')
	parser.add_argument('--eval_mode', action="store_true", default=False,
						help='enable for evaluation mode')
	parser.add_argument('--train_mask', action="store_true", default=False,
						help='enable for evaluation mode')
	parser.add_argument('--det_train_json', type=str, default="/path/to/train/json",
						help='absolute path of json file of the detection train dataset')
	parser.add_argument('--segm_train_json', type=str, default="/path/to/train/json",
						help='absolute path of json file of the object separation train dataset')
	parser.add_argument('--test_json', type=str, default="/path/to/test/json",
						help='absolute path of json file of the test dataset')
	parser.add_argument('--test_csv', type=str, default="/path/to/test/csv",
						help='absolute path of csv file of the eval dataset')
	parser.add_argument('--save_path', type=str, default="/path/to/save/weights",
						help='absolute path of the directory to save model weights')
	parser.add_argument('--segm_wts_path', type=str, default="/path/to/load/segm/weights",
						help='absolute path of the directory to load segmentation model weights from')
	parser.add_argument('--det_wts_path', type=str, default="/path/to/load/det/weights",
						help='absolute path of the directory to load detection model weights from')
	parser.add_argument('--wts_path', type=str, default="/path/to/load/weights",
						help='absolute path of the directory to load complete model weights from')
	parser.add_argument('--print_freq', type=int, default=20,
						help='frequency of printing logs')
	
	parser.add_argument('--pretrain_iter', type=int, default=0,
						help='iterations to train the detection section before switching')
	parser.add_argument('--switch_iter', type=int, default=0,
						help='iterations to train both models between switches')
	parser.add_argument('--max_iter', type=int, default=200000,
						help='maximum number of iterations before we stop')
	parser.add_argument('--save_iter', type=int, default=2500,
						help='iterations to save checkpoint weights')
	parser.add_argument('--det_resume_iter', type=int, default=0,
						help='iterations to resume training the detection section')
	parser.add_argument('--segm_resume_iter', type=int, default=0,
						help='iterations to resume training the segmentation section')
	parser.add_argument('--eval_image', type=str, default="/path/to/test/image",
						help='absolute path of the test image')
	parser.add_argument('--load_best', action="store_true", default=False,
						help='enable for to load best weights')
	parser.add_argument('--tta', action="store_true", default=False,
						help='enable prediction with TTA')

	args = parser.parse_args()
	
	cfg = setup(args)

	if args.eval_mode:
		do_eval(cfg, args.eval_image, args.tta)
	else:
		model = build_model(cfg)
		register_dataset(cfg)
		do_train(cfg, model, resume=True)