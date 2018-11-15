import argparse
import os
import shutil
import time
import random
import numpy as np
import math
import sys
from pytorch_classification.utils import Bar, AverageMeter

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from utils import *
from model import GolaiZero as golaiZero

class NNetWrapper():
    def __init__(self):
        self.nnet = golaiZero()
        self.program_x = PROGRAM_WIDTH
        self.program_y = PROGRAM_HEIGHT
        self.action_size = VOCAB_LEN

        if CUDA:
            self.nnet.cuda()
            #self.nnet = torch.nn.DataParallel(self.nnet, device_ids=[0, 1])

    def train(self, examples):
        """
        examples: list of examples, each example is of form (program, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters())

        for epoch in range(EPOCHS):
            print('EPOCH ::: ' + str(epoch+1))
            self.nnet.train()
            data_time = AverageMeter()
            batch_time = AverageMeter()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            end = time.time()

            bar = Bar('Training Net', max=int(len(examples)/BATCH_SIZE))
            batch_idx = 0

            while batch_idx < int(len(examples)/BATCH_SIZE):
                sample_ids = np.random.randint(len(examples), size=BATCH_SIZE)
                programs, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                programs = torch.FloatTensor(np.array(programs).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                if CUDA:
                    programs, target_pis, target_vs = programs.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()
                programs, target_pis, target_vs = Variable(programs), Variable(target_pis), Variable(target_vs)

                # measure data loading time
                data_time.update(time.time() - end)

                # compute output
                out_pi, out_v = self.nnet(programs)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.data.item(), programs.size(0))
                v_losses.update(l_v.data.item(), programs.size(0))

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                batch_idx += 1

                # plot progress
                bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss_pi: {lpi:.4f} | Loss_v: {lv:.3f}'.format(
                            batch=batch_idx,
                            size=int(len(examples)/BATCH_SIZE),
                            data=data_time.avg,
                            bt=batch_time.avg,
                            total=bar.elapsed_td,
                            eta=bar.eta_td,
                            lpi=pi_losses.avg,
                            lv=v_losses.avg,
                            )
                bar.next()
            bar.finish()


    def predict(self, program, gpuId):
        """
        program: np array with program
        """       
        with torch.cuda.device(int(gpuId)):
            # timing
            start = time.time()

            # preparing input
            program = torch.FloatTensor(program.astype(np.float64))
            if CUDA: program = program.contiguous().cuda()
            with torch.no_grad():
                program = Variable(program)

                #print(program.size())
                #program = program.view(1, 2, self.program_x, self.program_y)

                self.nnet.eval()
                pi, v = self.nnet(program)

                #print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets*outputs)/targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets-outputs.view(-1))**2)/targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({'state_dict' : self.nnet.state_dict(),}, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path {}".format(filepath))
        checkpoint = torch.load(filepath)
        self.nnet.load_state_dict(checkpoint['state_dict'])
