"""
The default lightning trainer for KNet.
"""

from pytorch_lightning import Trainer
from src.KNet.knet import *

model = KNet()

trainer = Trainer()  # gpus=[0] for GPU
trainer.fit(model)
# trainer.test(model)

# view tensorflow logs
print(f'View tensorboard logs by running\ntensorboard --logdir {os.getcwd()}')
print('and going to http://localhost:6006 on your browser')
