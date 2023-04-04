from models.PAVS360 import PAVS_V7

ENVIRONMENT = "local"
if ENVIRONMENT == "colab":
    # Google Colab ENV
    ENV_ROOT = "/content/AVS360_patched/"
    DATASETS_ROOT = "/content/"
    OUTPUT_ROOT = "/content/drive/MyDrive/MRGCV/TFM/"
    DATASET_CACHE_DIR = "/content/cached_data/"
else:
    # Local ENV
    ENV_ROOT = ""
    DATASETS_ROOT = "../../"
    OUTPUT_ROOT = "F:/Unizar/MRGCV/TFM/"
    DATASET_CACHE_DIR = OUTPUT_ROOT+"cached_data/"

SAVED_MODELS_ROOT = OUTPUT_ROOT+"saved_models/"
OUTPUTS_ROOT = OUTPUT_ROOT+"output/"
MODEL_NAME = "colab_batches_15"

# Datasets
class Dataset_Config:
    def __init__(self):
        self.VIDEO_ROOT_FOLDER = DATASETS_ROOT + self.DATASET_NAME+ '_frames_wav/'
        self.VIDEO_TRAIN_FOLDER = self.VIDEO_ROOT_FOLDER + 'TRAIN/'
        self.VIDEO_VALIDATION_FOLDER = self.VIDEO_ROOT_FOLDER + 'VALIDATION/'
        self.VIDEO_TEST_FOLDER = self.VIDEO_ROOT_FOLDER + 'TEST/'

        self.VIDEO_AEM_FOLDER = DATASETS_ROOT + self.DATASET_NAME + '_AEM/'
        self.VIDEO_SALIENCY_GT_FOLDER = DATASETS_ROOT + self.DATASET_NAME + '_saliency_GT_ambix/'
        self.VIDEO_FIXATION_GT_FOLDER = DATASETS_ROOT + self.DATASET_NAME + '_fixations_GT_ambix/'

        self.HAS_AEM = True

class Dataset_Config_2(Dataset_Config):
    def __init__(self, has_aem = True):
        super(Dataset_Config_2, self).__init__()
        self.VIDEO_ROOT_FOLDER = "F:/Unizar/MRGCV/TFM/datasets/" + self.DATASET_NAME+ '/frames_wav/'
        self.VIDEO_TRAIN_FOLDER = self.VIDEO_ROOT_FOLDER + 'TRAIN/'
        self.VIDEO_VALIDATION_FOLDER = self.VIDEO_ROOT_FOLDER + 'VALIDATION/'
        self.VIDEO_TEST_FOLDER = self.VIDEO_ROOT_FOLDER + 'TEST/'

        self.VIDEO_AEM_FOLDER = "F:/Unizar/MRGCV/TFM/datasets/" + self.DATASET_NAME + '/AEM/'
        self.VIDEO_SALIENCY_GT_FOLDER = "F:/Unizar/MRGCV/TFM/datasets/" + self.DATASET_NAME + '/saliency_GT/'
        self.VIDEO_FIXATION_GT_FOLDER = "F:/Unizar/MRGCV/TFM/datasets/" + self.DATASET_NAME + '/fixations_GT/'
        self.HAS_AEM = has_aem

class GAZE2018(Dataset_Config_2):
    def __init__(self):
        self.DATASET_NAME = "GAZE2018"
        super(GAZE2018, self).__init__(False)

class ASOD60k(Dataset_Config_2):
    def __init__(self):
        self.DATASET_NAME = "ASOD60k"
        super(ASOD60k, self).__init__()

class ICME2020(Dataset_Config):
    def __init__(self):
        self.DATASET_NAME = "360_Audio_Visual_ICMEW2020"
        super(ICME2020, self).__init__()

DATASETS = [ASOD60k()]
DATASET_CACHE_RESULTS = True
DATASET_SKIP_AUGMENTATIONS=True
DATASET_MAX_CACHED_RESULTS = -1 # If -1 all dataset will be cached,
# make sure to clear the cache directory on each run if there are any changes made

MODELS = [PAVS_V7] # Models to evaluate

# TRAIN PARAMETERS
TRAIN_MODEL_PATH = None
RESUME_TRAINING_EPOCH = None
DEBUG_TRAINING_DATA = True
MAX_TRAIN_FRAME_NUM = 600
TRAIN_FRAME_STEP = 6
TRAIN_START_SECONDS_OFFSET = 2
TRAIN_BATCH_SIZE = 1
NUM_EPOCHS = 200
DEPTH = 6

# DATASET LOADER PARAMETERS
NORMALIZE_FRAMES = True
FRAME_WIDTH = 256
FRAME_HIGHT = 128

#TOWARDS_FRAME_WIDTH = 320
#TOWARDS_FRAME_HEIGHT = 256

AEM_WIDTH = FRAME_WIDTH
AEM_HIGHT = FRAME_HIGHT

GT_WIDTH = 256
GT_HIGHT = 128

# ECB weights path
ECB = ENV_ROOT+'ECB.png'

# PREDICTION PARAMETERS
TEST_START_SECONDS_OFFSET = 0
PREDICTION_OUTPUT_FOLDER = 'output_'+MODEL_NAME+'/'
ACCUMULATED_OUTPUT_FOLDER = 'accumulated_saliency_maps/'
# model weights to load for predictions
MODEL_PATH = SAVED_MODELS_ROOT+MODEL_NAME+'/'+MODEL_NAME+'_ep_best.pkl'

# Where to save model parameters
MODEL_OUTPUT_FOLDER = SAVED_MODELS_ROOT+'/'
LOSSES_OUTPUT_PATH = OUTPUT_ROOT+MODEL_NAME+'_losses.csv'
SCORES_OUTPUT_PATH = PREDICTION_OUTPUT_FOLDER+MODEL_NAME+'_score.csv'

# EVALUATION PARAMETERS
EVALUATION_FRAME_STEP = 5
EVALUATION_START_SECONDS_OFFSET = TRAIN_START_SECONDS_OFFSET
EVALUATION_FOLDERS = ['F:/Unizar/MRGCV/TFM/output/DANI_EQUI/ASOD60k']
#EVALUATION_FOLDERS = ['output_finetuning_AEM/', 'output_AEM_finetuned_skip_2s/']
