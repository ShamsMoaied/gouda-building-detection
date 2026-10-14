"""Mask R-CNN with a ResNet-50 FPN backbone, pre-trained on COCO, adapted to one class (building)."""

from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def build_model(pretrained=True):
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = maskrcnn_resnet50_fpn(
        weights=weights,
        # The chips are 256 px. Upscaling them to 512 inside the model makes small
        # terraced houses large enough for the anchors; 800 (the default) is too slow on a CPU.
        min_size=512,
        max_size=512,
        # Dense residential chips can hold 80+ buildings; the default cap of 100 is too tight.
        box_detections_per_img=200,
    )
    # Swap the COCO heads (91 classes) for new ones with 2 classes: background + building.
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    in_channels = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels, 256, 2)
    return model
