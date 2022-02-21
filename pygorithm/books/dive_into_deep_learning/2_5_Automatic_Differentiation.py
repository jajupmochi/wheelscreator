#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 10:57:36 2022

@author: ljia
"""

import torch


#%% 2.5.1. A Simple Example
### 2.5.1. 一个简单的例子

x = torch.arange(4.0)
print(x)

x.requires_grad_(True) # == x=torch.arange(4.0,requires_grad=True)
print(x.grad)

y = 2 * torch.dot(x, x)
print(y)

y.backward()
print(x.grad)

print(x.grad == 4 * x)

# PyTorch accumulates the gradient in default, we need to clear the previous
# values.
# 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值.
x.grad.zero_()
print(x.grad)
y = x.sum()
y.backward()
print(x.grad)


#%% 2.5.2. Backward for Non-Scalar Variables
### 2.5.2. 非标量变量的反向传播

# Invoking `backward` on a non-scalar requires passing in a `gradient` argument
# which specifies the gradient of the differentiated function w.r.t `self`.
# In our case, we simply want to sum the partial derivatives, so passing
# in a gradient of ones is appropriate.
# 对非标量调用backward需要传入一个gradient参数，该参数指定微分函数关于self的梯度。
# 在我们的例子中，我们只想求偏导数的和，所以传递一个1的梯度是合适的.

x.grad.zero_()
y = x * x
# == 等价于y.backward(torch.ones(len(x)))
y.sum().backward()
print(x.grad)