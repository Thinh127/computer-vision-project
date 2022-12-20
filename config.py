# %% [code]
import torch

GRADIENT_PENALTY_WEIGHT = 10

EPOCHS = 10
MODEL_DIR = 'models'
MODEL_C = MODEL_DIR + '/colorization.pt'
MODEL_D = MODEL_DIR + '/discriminator.pt'

EVAL_DIR = 'evals'

TRAIN_PATH = '/kaggle/input/places365-sample/train_sample.txt'
TEST_PATH = '/kaggle/input/places365-sample/test_sample.txt'

OUTPUT_PATH = '/kaggle/working/test_output'

BATCH_SIZE = 32
TEST_BATCH_SIZE = 16

# -1 for no log
CHECK_PER = 100

LR = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"