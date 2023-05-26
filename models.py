import torch
import torch.nn as nn
from torch.nn import functional as F

from typing import Optional, List, Dict, Tuple
import functools, pickle, logging
import cv2 as cv
import numpy as np

from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.encoders._preprocessing import preprocess_input
from segmentation_models_pytorch.decoders.deeplabv3.decoder import SeparableConv2d, ASPPConv, ASPPSeparableConv
from segmentation_models_pytorch.base import initialization as init
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.losses import TverskyLoss

from detectron2.modeling import ROI_MASK_HEAD_REGISTRY, BaseMaskRCNNHead, ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.meta_arch import META_ARCH_REGISTRY, GeneralizedRCNN
from detectron2.modeling.roi_heads.mask_head import mask_rcnn_inference
from detectron2.structures import ImageList, Instances, Boxes, PolygonMasks
from detectron2.layers import ShapeSpec, cat
from detectron2.config import global_cfg
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import setup_logger

setup_logger()
logger = logging.getLogger("detectron2")

def mask_rcnn_loss(pred_mask_logits: torch.Tensor, instances: List[Instances], vis_period: int = 0):

	cls_agnostic_mask = pred_mask_logits.size(1) == 1
	total_num_masks = pred_mask_logits.size(0)
	mask_side_len = pred_mask_logits.size(2)
	assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"
	
	gt_classes = []
	gt_masks = []
	for instances_per_image in instances:
		if len(instances_per_image) == 0:
			continue
		if not cls_agnostic_mask:
			gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
			gt_classes.append(gt_classes_per_image)
		try:
			gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
				instances_per_image.proposal_boxes.tensor, mask_side_len
			).to(device=pred_mask_logits.device)
		except:
			gt_masks_per_image = torch.zeros((len(instances_per_image), mask_side_len, mask_side_len), 
										dtype=torch.bool, device=pred_mask_logits.device)
		gt_masks.append(gt_masks_per_image)
	
	if len(gt_masks) == 0:
		return {"loss_mask_bce" : pred_mask_logits.sum() * 0, "loss_mask_dice" : pred_mask_logits.sum() * 0}
	
	gt_masks = cat(gt_masks, dim=0)
	
	if cls_agnostic_mask:
		pred_mask_logits = pred_mask_logits[:, 0]
	else:
		indices = torch.arange(total_num_masks)
		gt_classes = cat(gt_classes, dim=0)
		pred_mask_logits = pred_mask_logits[indices, gt_classes]
	
	if gt_masks.dtype == torch.bool:
		gt_masks_bool = gt_masks
	else:
		gt_masks_bool = gt_masks > 0.5
	gt_masks = gt_masks.to(dtype=torch.float32)
	
	weights = []
	pred_masks_bool = torch.sigmoid(pred_mask_logits) > 0.5
	for enum, (gtm, pm) in enumerate(zip(gt_masks_bool, pred_masks_bool)):
		if gtm.sum() > 0:
			pwt = mask_side_len**2 / gtm.sum()
			weights.append(torch.where(gtm == True, pwt, torch.ones(1).to(device=pred_mask_logits.device)))
		else:
			weights.append(torch.ones_like(pred_mask_logits[0]))
	
	weights = torch.stack(weights, dim=0).to(device=pred_mask_logits.device)
 
	bce_loss = F.binary_cross_entropy_with_logits(pred_mask_logits, 
												gt_masks, 
												reduction="mean", 
												weight=weights)
	dice_loss = TverskyLoss(mode="binary", alpha=0.5, beta=0.5)(pred_mask_logits, gt_masks)
	return {"loss_mask_bce" : bce_loss, "loss_mask_dice" : dice_loss}

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates, separable=False):
        super(ASPP, self).__init__()
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        ASPPConvModule = ASPPConv if not separable else ASPPSeparableConv

        modules.append(ASPPConvModule(in_channels, out_channels, rate1))
        modules.append(ASPPConvModule(in_channels, out_channels, rate2))
        modules.append(ASPPConvModule(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU(),
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class DeepLabV3PlusDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        out_channels=256,
        atrous_rates=(4, 8, 12),
        output_stride=16,
    ):
        super().__init__()
        if output_stride not in {8, 16}:
            raise ValueError("Output stride should be 8 or 16, got {}.".format(output_stride))

        self.out_channels = out_channels
        self.output_stride = output_stride

        self.aspp = nn.Sequential(
            ASPP(encoder_channels[-1], out_channels, atrous_rates, separable=True),
            SeparableConv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        scale_factor = 8 
        self.up = nn.UpsamplingBilinear2d(scale_factor=scale_factor)

        highres_in_channels = 256 
        highres_out_channels = 48
        self.block1 = nn.Sequential(
            nn.Conv2d(highres_in_channels, highres_out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(highres_out_channels),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            SeparableConv2d(
                highres_out_channels + out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, *features):
        aspp_features = self.aspp(features[-1])
        aspp_features = self.up(aspp_features)
        high_res_features = self.block1(features[0])
        concat_features = torch.cat([aspp_features, high_res_features], dim=1)
        fused_features = self.block2(concat_features)
        return fused_features
		
@ROI_MASK_HEAD_REGISTRY.register()
class DeepLabV3Plus(BaseMaskRCNNHead):

    def __init__(self, cfg, input_shape: ShapeSpec):
        super(DeepLabV3Plus, self).__init__()

        self.encoder = get_encoder(
            name = "resnet18",
            in_channels = input_shape.channels,
            depth= 3 ,
            weights = "imagenet",
        )

        self.decoder = DeepLabV3PlusDecoder(
            encoder_channels = self.encoder.out_channels,
            out_channels = 256,
            atrous_rates = (12, 24, 36),
            output_stride = 16,
        )

        self.segmentation_head = SegmentationHead(
            in_channels = self.decoder.out_channels,
            out_channels = cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            activation = None,
            kernel_size = 1,
            upsampling = 2,
        )

        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)

    def layers(self, x):
        feats = self.encoder(x)
        x = self.decoder(*feats)
        x = self.segmentation_head(x)

        return x

    def forward(self, x, instances: List[Instances]):
        x = self.layers(x)
        if self.training:
            return mask_rcnn_loss(x, instances)
        else:
            mask_rcnn_inference(x, instances)
            return instances
			
@ROI_HEADS_REGISTRY.register()
class ModROIHeads(StandardROIHeads):

	def forward(
		self,
		images: ImageList,
		features: Dict[str, torch.Tensor],
		proposals: List[Instances],
		targets: Optional[List[Instances]] = None,
	) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
		
		del images            
	
		if self.training:
			assert targets, "'targets' argument is required during training"
			proposals = self.label_and_sample_proposals(proposals, targets)
		del targets
	
		if self.training:
			losses, proposals = self._forward_box(features, proposals)
	
			if global_cfg.GLOBAL.TRAIN_MASK:
				losses.update(self._forward_mask(features, proposals))
	
			storage = get_event_storage()
			if storage.iter % global_cfg.GLOBAL.PRINT_FREQ == 0:
				logger.info("Cumulative negative count: {}".format(global_cfg.GLOBAL.HN_COUNT))
				logger.info("Cumulative positive count: {}".format(global_cfg.GLOBAL.HP_COUNT))
				logger.info("Threshold: {}".format(global_cfg.GLOBAL.THRESHOLD))
				global_cfg.GLOBAL.HN_COUNT = [0] * 5
				global_cfg.GLOBAL.HP_COUNT = [0] * 5
			
			return proposals, losses
		else:
			pred_instances = self._forward_box(features, proposals)
			pred_instances = self.forward_with_given_boxes(features, pred_instances)
			return pred_instances, {}
		
	def _forward_mask(self, features: Dict[str, torch.Tensor], instances: List[Instances]):
		if not self.mask_on:
			return {} if self.training else instances
		if self.mask_pooler is not None:
			features = [features[f] for f in self.mask_in_features]
			boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
			features = self.mask_pooler(features, boxes)
		else:
			features = {f: features[f] for f in self.mask_in_features}
		return self.mask_head(features, instances)
	
	def _forward_box(self, features: Dict[str, torch.Tensor], proposals: List[Instances]):
		features = [features[f] for f in self.box_in_features]
		box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
		box_features = self.box_head(box_features)
		predictions = self.box_predictor(box_features)
		
		del box_features
		
		if self.training:
			losses = self.box_predictor.losses(predictions, proposals) if not global_cfg.GLOBAL.TRAIN_MASK else {}
			if self.train_on_pred_boxes:
				with torch.no_grad():
					pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
						predictions, proposals
					)
					for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
						proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
		
					pred_probs = self.box_predictor.predict_probs(predictions, proposals)
					filtered_proposals = []
					for enum, (proposal_per_img, pred_probs_per_img) in enumerate(zip(proposals, pred_probs)):
						gt_classes = proposal_per_img.gt_classes.clone()
						pr_classes = torch.argmax(pred_probs_per_img[:, :-1], 1)
						pr_scores = pred_probs_per_img[:, :-1][torch.arange(len(pred_probs_per_img)), pr_classes]
						if len(torch.unique(gt_classes)) == 1 and torch.unique(gt_classes)[0] == 5:
							selection_idxs = torch.argwhere(pr_scores > global_cfg.GLOBAL.THRESHOLD).squeeze(1)
							for si in selection_idxs:
								global_cfg.GLOBAL.HN_COUNT[pr_classes[si.item()]] += 1
						else:
							selection_idxs = torch.argwhere(gt_classes != 5).squeeze(1)
							for si in selection_idxs:
								global_cfg.GLOBAL.HP_COUNT[pr_classes[si.item()]] += 1
						proposal_per_img.gt_classes = torch.where(gt_classes == 5, pr_classes, gt_classes)
						filtered_proposals.append(proposal_per_img[selection_idxs])
		
			return losses, filtered_proposals
		else:
			pred_instances, _ = self.box_predictor.inference(predictions, proposals)
			return pred_instances
		
@META_ARCH_REGISTRY.register()
class ModGeneralizedRCNN(GeneralizedRCNN):

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)
        
        losses = {}
        losses.update(detector_losses)
        if not global_cfg.GLOBAL.TRAIN_MASK:
            losses.update(proposal_losses)
        return losses