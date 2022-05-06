# original code from https://github.com/hila-chefer/Transformer-Explainability/blob/main/modules/layers_ours.py

import torch
from torch import nn
import torch.nn.functional as F

__all__ = ['forward_hook', 'Clone', 'Add', 'Cat', 'ReLU', 'GELU', 'Dropout', 'BatchNorm2d', 'Linear', 'MaxPool2d',
           'AdaptiveAvgPool2d', 'AvgPool2d', 'Conv2d', 'Sequential', 'safe_divide', 'Einsum', 'Softmax', 'IndexSelect',
           'LayerNorm', 'AddEye']


def safe_divide(a, b):
    den = b.clamp(min=1e-9) + b.clamp(max=1e-9)
    den = den + den.eq(0).type(den.type()) * 1e-9
    return a / den * b.ne(0).type(b.type())


def forward_hook(self, inp, output):
    if type(inp[0]) in (list, tuple):
        self.x = []
        for i in inp[0]:
            x = i.detach()
            x.requires_grad = True
            self.x.append(x)
    else:
        self.x = inp[0].detach()
        self.x.requires_grad = True

    self.y = output


def backward_hook(self, grad_input, grad_output):
    self.grad_input = grad_input
    self.grad_output = grad_output


class RelProp(nn.Module):
    """
    Rel prop basic class
    """
    def __init__(self):
        super().__init__()
        # if not self.training:
        self.register_forward_hook(forward_hook)

    def gradprop(self, z, x, s):
        c = torch.autograd.grad(z, x, s, retain_graph=True)
        return c

    def relprop(self, r, alpha):
        _ = alpha
        return r


class RelPropSimple(RelProp):
    """
    Rel prop simple class
    """
    def relprop(self, r, alpha):
        z = self.forward(self.x)
        s = safe_divide(r, z)
        c = self.gradprop(z, self.x, s)

        if not torch.is_tensor(self.x):
            outputs = [self.x[0] * c[0], self.x[1] * c[1]]
        else:
            outputs = self.x * (c[0])
        return outputs


class AddEye(RelPropSimple):
    """
    Add eye
    """
    # input of shape B, C, seq_len, seq_len
    def forward(self, inp):
        return inp + torch.eye(inp.shape[2]).expand_as(inp).to(inp.device)


class ReLU(nn.ReLU, RelProp):
    pass


class GELU(nn.GELU, RelProp):
    pass


class Softmax(nn.Softmax, RelProp):
    pass


class LayerNorm(nn.LayerNorm, RelProp):
    pass


class Dropout(nn.Dropout, RelProp):
    pass


class MaxPool2d(nn.MaxPool2d, RelPropSimple):
    pass


class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d, RelPropSimple):
    pass


class AvgPool2d(nn.AvgPool2d, RelPropSimple):
    pass


class Add(RelPropSimple):
    """
    Add with rel prop
    """
    def forward(self, inputs):
        return torch.add(*inputs)

    def relprop(self, r, alpha):
        z = self.forward(self.x)
        s = safe_divide(r, z)
        c = self.gradprop(z, self.x, s)

        a = self.x[0] * c[0]
        b = self.x[1] * c[1]

        a_sum = a.sum()
        b_sum = b.sum()

        a_fact = safe_divide(a_sum.abs(), a_sum.abs() + b_sum.abs()) * r.sum()
        b_fact = safe_divide(b_sum.abs(), a_sum.abs() + b_sum.abs()) * r.sum()

        a = a * safe_divide(a_fact, a.sum())
        b = b * safe_divide(b_fact, b.sum())

        outputs = [a, b]

        return outputs


class Einsum(RelPropSimple):
    """
    einsum with rel prop
    """
    def __init__(self, equation):
        super().__init__()
        self.equation = equation

    def forward(self, *operands):
        return torch.einsum(self.equation, *operands)


class IndexSelect(RelProp):
    """
    IndexSelect with rel prop
    """
    def forward(self, inputs, dim, indices):
        self.__setattr__('dim', dim)
        self.__setattr__('indices', indices)

        return torch.index_select(inputs, dim, indices)

    def relprop(self, r, alpha):
        z = self.forward(self.x, self.dim, self.indices)
        s = safe_divide(r, z)
        c = self.gradprop(z, self.x, s)

        if not torch.is_tensor(self.x):
            outputs = [self.x[0] * c[0], self.x[1] * c[1]]
        else:
            outputs = self.x * (c[0])
        return outputs


class Clone(RelProp):
    """
    Clone with rel prop
    """
    def forward(self, inp, num):
        self.__setattr__('num', num)
        outputs = []
        for _ in range(num):
            outputs.append(inp)

        return outputs

    def relprop(self, r, alpha):
        z = []
        for _ in range(self.num):
            z.append(self.x)
        s = [safe_divide(r, z) for r, z in zip(r, z)]
        c = self.gradprop(z, self.x, s)[0]

        r = self.x * c

        return r


class Cat(RelProp):
    """
    Cat with rel prop
    """
    def forward(self, inputs, dim):
        self.__setattr__('dim', dim)
        return torch.cat(inputs, dim)

    def relprop(self, r, alpha):
        z = self.forward(self.x, self.dim)
        s = safe_divide(r, z)
        c = self.gradprop(z, self.x, s)

        outputs = []
        for x, c1 in zip(self.x, c):
            outputs.append(x * c1)

        return outputs


class Sequential(nn.Sequential):
    """
    Sequential with rel prop
    """
    def relprop(self, r, alpha):
        for m in reversed(self._modules.values()):
            r = m.relprop(r, alpha)
        return r


class BatchNorm2d(nn.BatchNorm2d, RelProp):
    """
    BatchNorm2d with rel prop
    """
    def relprop(self, r, alpha):
        x = self.x
        beta = 1 - alpha
        _ = beta
        weight = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3) / (
            (self.running_var.unsqueeze(0).unsqueeze(2).unsqueeze(3).pow(2) + self.eps).pow(0.5))
        z = x * weight + 1e-9
        s = r / z
        ca = s * weight
        r = self.x * ca
        return r


class Linear(nn.Linear, RelProp):
    """
    Linear with rel prop
    """
    def relprop(self, r, alpha):
        beta = alpha - 1
        pw = torch.clamp(self.weight, min=0)
        nw = torch.clamp(self.weight, max=0)
        px = torch.clamp(self.x, min=0)
        nx = torch.clamp(self.x, max=0)

        def f(w1, w2, x1, x2):
            z1 = F.linear(x1, w1)
            z2 = F.linear(x2, w2)
            s1 = safe_divide(r, z1 + z2)
            s2 = safe_divide(r, z1 + z2)
            c1 = x1 * torch.autograd.grad(z1, x1, s1)[0]
            c2 = x2 * torch.autograd.grad(z2, x2, s2)[0]

            return c1 + c2

        activator_relevances = f(pw, nw, px, nx)
        inhibitor_relevances = f(nw, pw, px, nx)

        r = alpha * activator_relevances - beta * inhibitor_relevances

        return r


class Conv2d(nn.Conv2d, RelProp):
    """
    Conv2d with rel prop
    """
    def gradprop2(self, dy, weight):
        z = self.forward(self.x)

        output_padding = self.x.size()[2] - (
                (z.size()[2] - 1) * self.stride[0] - 2 * self.padding[0] + self.kernel_size[0])

        return F.conv_transpose2d(dy, weight, stride=self.stride, padding=self.padding, output_padding=output_padding)

    def relprop(self, r, alpha):
        if self.x.shape[1] == 3:
            pw = torch.clamp(self.weight, min=0)
            nw = torch.clamp(self.weight, max=0)
            x = self.x
            l = self.x * 0 + \
                torch.min(torch.min(torch.min(self.x, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3,
                          keepdim=True)[0]
            h = self.x * 0 + \
                torch.max(torch.max(torch.max(self.x, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3,
                          keepdim=True)[0]
            za = torch.conv2d(x, self.weight, bias=None, stride=self.stride, padding=self.padding) - \
                 torch.conv2d(l, pw, bias=None, stride=self.stride, padding=self.padding) - \
                 torch.conv2d(h, nw, bias=None, stride=self.stride, padding=self.padding) + 1e-9

            s = r / za
            c = x * self.gradprop2(s, self.weight) - l * self.gradprop2(s, pw) - h * self.gradprop2(s, nw)
            r = c
        else:
            beta = alpha - 1
            pw = torch.clamp(self.weight, min=0)
            nw = torch.clamp(self.weight, max=0)
            px = torch.clamp(self.x, min=0)
            nx = torch.clamp(self.x, max=0)

            def f(w1, w2, x1, x2):
                z1 = F.conv2d(x1, w1, bias=None, stride=self.stride, padding=self.padding)
                z2 = F.conv2d(x2, w2, bias=None, stride=self.stride, padding=self.padding)
                s1 = safe_divide(r, z1)
                s2 = safe_divide(r, z2)
                c1 = x1 * self.gradprop(z1, x1, s1)[0]
                c2 = x2 * self.gradprop(z2, x2, s2)[0]
                return c1 + c2

            activator_relevances = f(pw, nw, px, nx)
            inhibitor_relevances = f(nw, pw, px, nx)

            r = alpha * activator_relevances - beta * inhibitor_relevances
        return r
