from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import pandas as pd
from model import WhaleEfficientNet
from dataloader import WhaleTestDataset
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

TEST_IMAGE_DIRS = '../Competition/Jungle/test_features/'
SUBMISSION_FORMAT = '..Competition/Jungle/submission_format.csv'
LABEL_MAP = ['antelope_duiker', 'bird', 'blank', 'civet_genet', 'hog', 'leopard', 'monkey_prosimian', 'rodent']

version_name = 'version_0'
checkpoint_path = f'lightning_logs/{version_name}/checkpoints/best_weight.ckpt'
img_size = 128

test = pd.read_csv('../Competition/Jungle/test_features.csv')

test_dataset = WhaleTestDataset(path=TEST_IMAGE_DIRS,
                                image_ids=test.id.values,
                                img_size=img_size)
test_loader = DataLoader(test_dataset,
                         batch_size=512)

preds = np.empty((0, 8))

model = WhaleEfficientNet.load_from_checkpoint(checkpoint_path=checkpoint_path).to('cuda')
model.eval()
model.cuda()
model.freeze()

log_loss_val = 0

for batch in test_loader:
    input = batch['x']
    input = input.cuda()

    output = model(input)

    for out in output:
        res = F.softmax(out, dim=0).detach().to('cpu').numpy()

        # for idx, dump in enumerate(res):
        #     res[idx] = '{:.15f}'.format(res[idx])

        preds = np.append(preds, [res], axis=0)

        log_loss_val += np.argmax(res)

    # output = torch.argmax(output, dim=1)

submission = pd.read_csv(SUBMISSION_FORMAT)

for idx, pred in enumerate(preds):
    submission.iloc[idx, 1:] = pred

submission.to_csv(f'submission/{version_name}%_submission.csv', index=False)
