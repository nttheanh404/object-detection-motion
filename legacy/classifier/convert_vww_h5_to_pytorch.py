#!/usr/bin/env python3
from pathlib import Path
import json, math, argparse
import h5py, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F

class SamePad2d(nn.Module):
    def __init__(self,kernel_size,stride):
        super().__init__(); self.k=kernel_size if isinstance(kernel_size,tuple) else (kernel_size,kernel_size); self.s=stride if isinstance(stride,tuple) else (stride,stride)
    def forward(self,x):
        ih,iw=x.shape[-2:]; kh,kw=self.k; sh,sw=self.s
        oh=math.ceil(ih/sh); ow=math.ceil(iw/sw)
        ph=max((oh-1)*sh+kh-ih,0); pw=max((ow-1)*sw+kw-iw,0)
        return F.pad(x,[pw//2,pw-pw//2,ph//2,ph-ph//2])

class VWWMobileNetV1(nn.Module):
    def __init__(self):
        super().__init__()
        layers=[]; in_ch=3
        # (type, out_ch, stride) after first conv and 13 depthwise-pointwise blocks
        layers += [('conv','conv2d',8,2)]
        specs=[(8,16,1),(16,32,2),(32,32,1),(32,64,2),(64,64,1),(64,128,2),(128,128,1),(128,128,1),(128,128,1),(128,128,1),(128,128,1),(128,256,2),(256,256,1)]
        self.blocks=nn.ModuleList(); self.names=[]
        conv_idx=0; dw_idx=0; bn_idx=0
        # first conv block
        self.blocks.append(nn.Sequential(SamePad2d(3,2), nn.Conv2d(3,8,3,2,0,bias=True), nn.BatchNorm2d(8,eps=1e-3,momentum=0.99), nn.ReLU(inplace=False)))
        self.names.append(('conv2d','batch_normalization',None))
        for i,(inch,outch,stride) in enumerate(specs):
            dw_name='depthwise_conv2d' + ('' if i==0 else f'_{i}')
            pw_name='conv2d_'+str(i+1)
            bn_dw='batch_normalization_'+str(1+2*i)
            bn_pw='batch_normalization_'+str(2+2*i)
            self.blocks.append(nn.Sequential(
                SamePad2d(3,stride), nn.Conv2d(inch,inch,3,stride,0,groups=inch,bias=True), nn.BatchNorm2d(inch,eps=1e-3,momentum=0.99), nn.ReLU(inplace=False),
                nn.Conv2d(inch,outch,1,1,0,bias=True), nn.BatchNorm2d(outch,eps=1e-3,momentum=0.99), nn.ReLU(inplace=False)
            ))
            self.names.append((dw_name,bn_dw,pw_name,bn_pw))
        self.pool=nn.AvgPool2d(3,3)
        self.fc=nn.Linear(256,2,bias=True)
    def forward(self,x):
        for b in self.blocks: x=b(x)
        x=self.pool(x); x=torch.flatten(x,1); return self.fc(x)

def get_arr(f,layer,ds):
    return np.array(f['model_weights'][layer][layer][ds])

def load_bn(f,bn,name):
    gamma=get_arr(f,name,'gamma:0'); beta=get_arr(f,name,'beta:0'); mean=get_arr(f,name,'moving_mean:0'); var=get_arr(f,name,'moving_variance:0')
    bn.weight.data.copy_(torch.from_numpy(gamma)); bn.bias.data.copy_(torch.from_numpy(beta)); bn.running_mean.copy_(torch.from_numpy(mean)); bn.running_var.copy_(torch.from_numpy(var))

def load_conv(f,conv,name,depthwise=False):
    if depthwise:
        w=get_arr(f,name,'depthwise_kernel:0') # h,w,in,1
        w=np.transpose(w,(2,3,0,1)) # in,1,h,w
    else:
        w=get_arr(f,name,'kernel:0') # h,w,in,out
        w=np.transpose(w,(3,2,0,1))
    b=get_arr(f,name,'bias:0')
    conv.weight.data.copy_(torch.from_numpy(w)); conv.bias.data.copy_(torch.from_numpy(b))

def convert(h5,out):
    model=VWWMobileNetV1().eval()
    with h5py.File(h5,'r') as f:
        # first block: [pad, conv, bn, relu]
        load_conv(f,model.blocks[0][1],'conv2d'); load_bn(f,model.blocks[0][2],'batch_normalization')
        for bi in range(1,len(model.blocks)):
            names=model.names[bi]; dw,bndw,pw,bnpw=names
            block=model.blocks[bi]
            load_conv(f,block[1],dw,depthwise=True); load_bn(f,block[2],bndw)
            load_conv(f,block[4],pw); load_bn(f,block[5],bnpw)
        w=get_arr(f,'dense','kernel:0') # in,out
        b=get_arr(f,'dense','bias:0')
        model.fc.weight.data.copy_(torch.from_numpy(w.T)); model.fc.bias.data.copy_(torch.from_numpy(b))
    out=Path(out); out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'model':model.state_dict(),'arch':'VWWMobileNetV1','params':sum(p.numel() for p in model.parameters())},out)
    print('saved',out,'params',sum(p.numel() for p in model.parameters()))

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--h5',required=True); ap.add_argument('--out',required=True); args=ap.parse_args(); convert(args.h5,args.out)
