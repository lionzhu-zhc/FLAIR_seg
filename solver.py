import torch
import torch.nn as nn
from torch.autograd import Variable as V
import sys

#-adjust lr-------------------------------------------------------------------------------------------------------------
def adjust_lr(optimizer, LR):
    for param_group in optimizer.param_groups:
        param_group['lr'] = LR


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
	    self.terminal = stream
	    self.log = open(filename, 'a')

    def write(self, message):
	    self.terminal.write(message)
	    self.log.write(message)

    def flush(self):
	    pass
