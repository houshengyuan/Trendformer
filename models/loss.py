#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
from torch import nn

class BCE_Trend_MSE_Period(nn.Module):
   def __init__(self,):
      super(BCE_Trend_MSE_Period, self).__init__()
      self.BCE = nn.BCELoss()
      self.MSE = nn.MSELoss()

   def forward(self, pred, true):
      pred_trend, pred_period = pred
      true_trend, true_period = true
      bce_loss = self.BCE(pred_trend, true_trend)
      mse_loss = self.MSE(pred_period, true_period)
      return bce_loss+mse_loss

class MSE_Trend_Period(nn.Module):
   def __init__(self,):
      super(MSE_Trend_Period, self).__init__()
      self.MSE = nn.MSELoss()

   def forward(self, pred, true):
      pred_trend, pred_period = pred
      true_trend, true_period = true
      mse_loss1 = self.MSE(pred_trend, true_trend)
      mse_loss2 = self.MSE(pred_period, true_period)
      return mse_loss1+mse_loss2