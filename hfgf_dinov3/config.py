"""Configuration for the released HFGF-DINOv3 model."""

from detectron2.config import CfgNode as CN, get_cfg
from detectron2.projects.deeplab import add_deeplab_config


def add_hfgf_dinov3_config(cfg):
    """Register the settings used by the retained training recipe."""
    cfg.MODEL.BACKBONE_WEIGHTS = ""
    cfg.INPUT.DATASET_MAPPER_NAME = "hfgf_dataset_mapper"
    cfg.INPUT.COLOR_AUG_SSD = True
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    cfg.INPUT.SIZE_DIVISIBILITY = 512
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1
    cfg.TEST.METRIC_WORKERS = 0

    cfg.MODEL.VITSPLIT = CN()
    split = cfg.MODEL.VITSPLIT
    split.BACKBONE_SIZE = "v3-h+"
    split.SELECT_LAYERS = [25, 26, 27, 28, 29, 30, 31]
    split.OUT_NUM = 9
    split.CHANNELS = 256
    split.FROZEN_DIM = 1280
    split.FUSION_OUT_DIM = 256
    split.TUNING_TYPE = "all"
    split.FREEZE_GATE = False
    split.MS_LAYERS = [28, 29, 30, 31]
    split.FUSION_MODE = "hfgf"
    split.HFGF_USE_MASK = True
    split.HFGF_GAIN = 1.0
    split.HFGF_GAMMA_INIT = 0.0
    split.HFGF_ENABLE_SPATIAL = True
    split.HFGF_ENABLE_FREQ = True
    split.USE_SPLIT_CHECKPOINT = False
    split.SPLIT_CHECKPOINT_SEGMENTS = 2

    cfg.MODEL.MASK_FORMER = CN()
    decoder = cfg.MODEL.MASK_FORMER
    decoder.DEEP_SUPERVISION = True
    decoder.NO_OBJECT_WEIGHT = 0.1
    decoder.CLASS_WEIGHT = 1.0
    decoder.MASK_WEIGHT = 5.0
    decoder.DICE_WEIGHT = 5.0
    decoder.BBOX_WEIGHT = 5.0
    decoder.GIOU_WEIGHT = 2.0
    decoder.BAPHF_WEIGHT = 0.1
    decoder.BAPHF_START_ITER = 2000
    decoder.BAPHF_HIGHPASS = 0.25
    decoder.BAPHF_BAND_KERNEL = 3
    decoder.NHEADS = 8
    decoder.DROPOUT = 0.0
    decoder.DIM_FEEDFORWARD = 2048
    decoder.DEC_LAYERS = 10
    decoder.PRE_NORM = False
    decoder.HIDDEN_DIM = 256
    decoder.NUM_OBJECT_QUERIES = 2
    decoder.USE_FEATURE_MASK_ATTN = True
    decoder.TRANSFORMER_IN_FEATURE = "multi_scale_pixel_decoder"
    decoder.ENFORCE_INPUT_PROJ = False
    decoder.TRANSFORMER_DECODER_NAME = "MultiScaleMaskedTransformerDecoder"
    decoder.TRAIN_NUM_POINTS = 12544
    decoder.OVERSAMPLE_RATIO = 3.0
    decoder.IMPORTANCE_SAMPLE_RATIO = 0.75
    decoder.TEST = CN()
    decoder.TEST.SEMANTIC_ON = True
    decoder.TEST.INSTANCE_ON = True
    decoder.TEST.DETECTIONS_PER_IMAGE = 4
    decoder.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False

    head = cfg.MODEL.SEM_SEG_HEAD
    head.MASK_DIM = 256
    head.TRANSFORMER_ENC_LAYERS = 6
    head.PIXEL_DECODER_NAME = "MSDeformAttnPixelDecoder"
    head.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    head.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    head.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8


def load_config(path, overrides=()):
    """Load a YAML recipe with Detectron2-style KEY VALUE overrides."""
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_hfgf_dinov3_config(cfg)
    cfg.merge_from_file(path)
    cfg.merge_from_list(list(overrides))
    cfg.freeze()
    return cfg
