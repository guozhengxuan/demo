import os
import torch

HOME = os.path.expanduser('~')

GPUS = torch.cuda.device_count()

BASE = HOME + '/torch_app_data'

MODEL = BASE + '/model'
BERT = BASE + '/bert'

DATASET = BASE + '/dataset'
SA_DATASET = DATASET + '/sa'
TC_DATASET = DATASET + '/tc'

SA_MODEL = MODEL + '/sa'
SA_BEST_MODEL = MODEL + '/sa_best'
TC_MODEL = MODEL + '/tc'

SA_TOTAL = SA_DATASET + '/total.csv'
SA_TRAIN = SA_DATASET + '/train.csv'
SA_TRAIN_CACHE = SA_DATASET + '/train_cache.pk'
SA_TEST = SA_DATASET + '/predict.csv'
SA_DEV = SA_DATASET + '/dev.csv'
