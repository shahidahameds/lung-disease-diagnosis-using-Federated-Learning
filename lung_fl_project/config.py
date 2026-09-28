import os

# Base paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# FL Configuration
NUM_HOSPITALS = 4
DEPARTMENTS_PER_HOSPITAL = 2
TOTAL_CLIENTS = NUM_HOSPITALS * DEPARTMENTS_PER_HOSPITAL
TEST_SPLIT = 0.2
RANDOM_SEED = 42

# Image config
IMAGE_SIZE = (75, 75)
BATCH_SIZE = 8
LEARNING_RATE = 0.0001

# Supported extensions
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

# FL & Experiment Configuration Defaults
NUM_ROUNDS = 3
LOCAL_EPOCHS = 1
CLIP_NORM = 1.0
NOISE_MULTIPLIER = 0.05
ANOMALY_THRESHOLD = 0.60
DEVICE = "cuda" # Default, scripts should auto-detect
USE_PRETRAINED_WEIGHTS = True
MAX_TRAIN_SAMPLES = None
MAX_TEST_SAMPLES = None
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
EXPERIMENT_MAX_TRAIN_SAMPLES_PER_CLIENT = 32
EXPERIMENT_MAX_TEST_SAMPLES = 128
FULL_DATASET_RUN = False

def detect_dataset_classes(data_dir=DATA_DIR):
	"""Return sorted class folders present in the predefined dataset splits."""
	classes = set()
	for split in ("train", "test", "val"):
		split_dir = os.path.join(data_dir, split)
		if not os.path.isdir(split_dir):
			continue
		classes.update(
			name for name in os.listdir(split_dir)
			if os.path.isdir(os.path.join(split_dir, name))
		)
	return sorted(classes)


# Derived from the folders that physically exist in the dataset.
NUM_CLASSES = len(detect_dataset_classes())
