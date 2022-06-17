import pytorch_lightning as pl
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from model import WhaleEfficientNet
from dataloader import WhaleDataModule

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

model_checkpoint = ModelCheckpoint(monitor='val_acc',
                                   verbose=True,
                                   save_last=True,
                                   save_top_k=5,
                                   mode='max',
                                   filename="{epoch}_{val_acc:.4f}")

early_stopping = EarlyStopping(
    monitor='val_acc',
    patience=10,
    verbose=True,
    mode='max'
)

img_size = 256
batch_size = 8

dm = WhaleDataModule(img_size, batch_size)
whale_model = WhaleEfficientNet()

trainer = pl.Trainer(gpus=-1,
                     max_epochs=100,
                     # auto_scale_batch_size="binsearch",
                     callbacks=[model_checkpoint, early_stopping])

# Run auto_scale_batch_size
# tuner = Tuner(trainer)
# batch_size = tuner.scale_batch_size(model=whale_model, mode='binsearch', init_val=1, max_trials=10, datamodule=dm)
# whale_model.hparams.batch_size = batch_size
# print(f'Batch Size : {batch_size}')

trainer.fit(model=whale_model,
            datamodule=dm)
            # datamodule=WhaleDataModule(batch_size))

trainer.save_checkpoint(f'{trainer.checkpoint_callback.dirpath}/best_weight.ckpt')
