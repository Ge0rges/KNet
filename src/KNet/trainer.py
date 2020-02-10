"""
The default lightning trainer for KNet.
"""

from pytorch_lightning import Trainer
from knet import *

model = KNet(784, [10, 10, 10])

trainer = Trainer()  # gpus=[0] for GPU
trainer.fit(model)
# trainer.test(model)

# view tensorflow logs
print(f'View tensorboard logs by running\ntensorboard --logdir {os.getcwd()}')
print('and going to http://localhost:6006 on your browser')
