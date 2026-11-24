from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLIT_FILE = DATA_DIR / "splits.csv"
CLEANING_REPORT = DATA_DIR / "cleaning_report.json"
STATS_FILE = DATA_DIR / "dataset_stats.json"

OUTPUT_DIR = ROOT / "outputs"
RUNS_DIR = OUTPUT_DIR / "runs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
BENCHMARK_DIR = OUTPUT_DIR / "benchmark"

KAGGLE_DATASET = "nirmalsankalana/sugarcane-leaf-disease-dataset"
CLASSES = ["Healthy", "Mosaic", "RedRot", "Rust", "Yellow"]
CLASS_NAMES_FR = {
    "Healthy": "Saine",
    "Mosaic": "Mosaïque",
    "RedRot": "Pourriture rouge",
    "Rust": "Rouille",
    "Yellow": "Jaunissement",
}

SEED = 42
CACHE_SIZE = 256
IMAGE_SIZE = 224
TEST_SIZE = 0.15
N_FOLDS = 5
NEAR_DUPLICATE_THRESHOLD = 0.90

BATCH_SIZE = 32
EPOCHS = 30
WARMUP_EPOCHS = 2
LR_NEW_LAYERS = 1e-3
LR_PRETRAINED = 1e-4
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1
DROP_RATE = 0.2

PRETRAINED_MODELS = {
    "efficientnetv2_s": "tf_efficientnetv2_s.in1k",
    "mobilenetv4_m": "mobilenetv4_conv_medium.e500_r224_in1k",
    "convnextv2_t": "convnextv2_tiny.fcmae_ft_in1k",
    "seresnet50": "seresnet50.ra2_in1k",
}
SCRATCH_MODEL = "scratch_cnn"

LAST_STAGE = {
    "efficientnetv2_s": ["blocks.5", "conv_head", "bn2"],
    "mobilenetv4_m": ["blocks.3", "blocks.4", "conv_head", "norm_head"],
    "convnextv2_t": ["stages.3", "head.norm"],
    "seresnet50": ["layer4"],
}

STRATEGIES = ["feature_extraction", "partial_finetuning", "full_finetuning"]
SCRATCH_STRATEGY = "from_scratch"

MODEL_LABELS = {
    "efficientnetv2_s": "EfficientNetV2-S",
    "mobilenetv4_m": "MobileNetV4-M",
    "convnextv2_t": "ConvNeXtV2-T",
    "seresnet50": "SE-ResNet-50",
    "scratch_cnn": "CNN from scratch",
    "fusion_imagenet": "Fusion (FE)",
    "fusion_finetuned": "Fusion (FFT)",
    "late_fusion": "Vote souple",
}

STRATEGY_LABELS = {
    "feature_extraction": "Extraction de caractéristiques",
    "partial_finetuning": "Fine-tuning partiel",
    "full_finetuning": "Fine-tuning complet",
    "from_scratch": "From scratch",
}
