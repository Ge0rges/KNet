from pytorch_lightning import Trainer
from knet import *

model = KNet(784, [10, 10, 10])

trainer = Trainer()
trainer.fit(model)
trainer.test(model)
