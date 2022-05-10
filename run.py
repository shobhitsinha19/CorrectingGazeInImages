from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from Dataset import Dataset
from model import model
from utils.config.train_options import TrainOptions




def test(opt):
    dataset = Dataset(opt)
    modelObj = model(dataset, opt)
    modelObj.build_test_model()
    modelObj.test()

def train(opt):
    dataset = Dataset(opt)
    modelObj = model(dataset, opt)
    modelObj.build_model()
    modelObj.train()

if __name__ == "__main__":
    opt = TrainOptions().parse()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_id)
    if opt.mode == "train":
        print("Running in training mode")
        train(opt)
    else:
        print("Running in testing mode")
        test(opt)
        