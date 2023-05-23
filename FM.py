import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder



class FM(nn.Module):

    def __init__(self, field_dim,
                 embed_dim, output_dim, reg=0.01):
        super(FM, self).__init__()
        self.w = nn.Linear(field_dim, output_dim, bias=True)
        self.v = nn.Parameter(torch.FloatTensor(output_dim, field_dim, embed_dim), requires_grad=True)
        self.field_dim = field_dim
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        nn.init.xavier_normal_(self.v)
        self.reg = reg


    def forward(self, input):
        """
        input shape: (num_item, field_dim)
        """
        linear_comb = self.w(input)

        repeated = input.repeat(self.output_dim, 1)
        result = repeated.reshape(self.output_dim, input.size()[0], input.size()[1])
        square_of_sum = torch.matmul(result, self.v)
        square_of_sum = torch.pow(square_of_sum, 2)
        first_term = torch.sum(square_of_sum, dim=2)

        square_of_v = torch.pow(self.v, 2)
        square_of_x = torch.pow(result, 2)
        sum_of_square = torch.matmul(square_of_x, square_of_v)
        second_term = self.reg * torch.sum(sum_of_square, dim=2)
        output = first_term - second_term
        p = linear_comb.squeeze(1) + output.transpose(0, 1)
        p = torch.softmax(p, dim=1)
        return p
