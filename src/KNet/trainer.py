from pytorch_lightning import Trainer
from src.KNet.knet import *

model = KNet()

trainer = Trainer()
trainer.fit(model)
#trainer.test(model)
