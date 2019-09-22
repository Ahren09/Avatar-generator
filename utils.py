# coding:utf8
from itertools import chain
import visdom
import torch
import time
import torchvision as tv
import numpy as np

class Visualizer():

    def __init__(self, env='main', **kwargs):
        self.vis = visdom.Visdom(env=env, use_incoming_socket=False, **kwargs)
        '''
        run `python -m visdom.server` before run in terminal
        '''

        # Define x-axis
        self.index = {}
        self.log_text = ''

    def reinit(self, env='main', **kwargs):
        self.vis = visdom.Visdom(env=env, use_incoming_socket=False, **kwargs)
        return self

    def plot_many(self, d):
        for key, value in d.items():
            self.plot(key, value)
    
    def plot(self, name, y):
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]), win=name, opts={'title':name}\
            , update=None if x==0 else 'append')
        self.index[name] = x+1

    def img(self, name, img_):
        if len(img_.size()) < 3:
            img_ = img_.cpu().unsqueeze(0)
        self.vis.image(img_.cpu(), win=name, opts=dict(title=name))

    def img_grid_many(self, d):
        for key, value in d.items():
            self.img_grid(key, value)

    def img_grid(self, name, input_3d):
        self.img(name, tv.utils.make_grid(
            input_3d.cpu()[0].unsqueeze(1).clamp(max=1, min=0)
        ))
    
    def log(self, info, win='log_text'):
        """
        self.log({'loss':1,'lr':0.0001})
        """

        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'),
            info=info))
        self.vis.text(self.log_text, win=win)

    def __getattr__(self, name):
        return getattr(self.vis, name)
    