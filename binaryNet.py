import torch
from torch.autograd import Function
import numpy as np
#import torch.nn.functional as F

class Binary(Function):
    # def __init__(self):
    #     super(Binary,self).__init__()

    @staticmethod
    def forward(self, input):
        output = torch.sign(input)
        self.save_for_backward(input)
        return output

    @staticmethod
    def backward(self, grad_output):
#        input = self.saved_tensors
#        grad_input = torch.zeros(grad_output.size()).type_as(grad_output)
#        print input
#        print torch.ge(input,torch.Tensor(1))
#        grad_input[torch.ge(input,torch.Tensor(1))]=0
#        grad_input[torch.le(input,torch.Tensor(-1))]=0
        grad_input = grad_output
        return grad_input


class Binary_W(Function):
    # def __init__(self):
    #     super(Binary_W,self).__init__()
    @staticmethod
    def forward(self, input, weight):
        new_weight = torch.sign(weight)
        new_input = torch.sign(input)
        self.save_for_backward(input, weight)
       # output = F.conv2d(new_input,new_weight)
        return  new_input, new_weight

    @staticmethod
    def backward(self, grad_input, grad_weight):
      #  input, weight = self.saved_tensors
       # print grad_input
        return grad_input, grad_weight


class Threshold(Function):
    # def __init__(self, th):
    #     super(Threshold,self).__init__()
    #     self.th = th
    #     self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    @staticmethod
    def forward(self, input, th):
        self.save_for_backward(input)
        m = torch.min(input)
        n = torch.max(input)
        r = torch.linspace(m, n, steps=th+2).to(input.device)
        for i in r[1:-1]: # excluding the smallest value, and the biggest
                output = input.clone()
                output[output< i]=torch.Tensor([-1]).to(input.device)
                output[output>=i]=torch.Tensor([1]).to(input.device)

                if i==r[1]:
                    out = output #torch.unsqueeze(output,0)
                else:
                    out = torch.cat([out,output],1) #torch.cat([out,torch.unsqueeze(output,0)],0)
        return out

    @staticmethod
    def backward(self, grad_output):
#        input = self.saved_tensors
#        grad_input = torch.zeros(grad_output.size()).type_as(grad_output)
#        print input
#        print torch.ge(input,torch.Tensor(1))
#        grad_input[torch.ge(input,torch.Tensor(1))]=0
#        grad_input[torch.le(input,torch.Tensor(-1))]=0
        grad_input = grad_output

        return grad_input, None
