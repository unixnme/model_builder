import torch.nn as nn
import json


class Reshape(nn.Module):
    def __init__(self, *out_shape):
        super().__init__()
        self.out_shape = (-1,) + out_shape

    def __str__(self):
        return str("Reshape" + str(self.out_shape[1:]))

    def __repr__(self):
        return self.__str__()

    def forward(self, x):
        return x.view(self.out_shape)


class custom_nn(object):
    @staticmethod
    def Reshape(*shape):
        return Reshape(*shape)


def build_from_file(filename:str):
    '''
    given json file
    :return: model
    '''
    model_def = json.load(open(filename, 'r'))
    return build(model_def)


def build(model_def:dict):
    if len(model_def) != 1:
        raise Exception('must have a single key')

    try:
        return build_sequential(model_def["sequential"])
    except:
        raise Exception("currently only sequential is supported")


def build_sequential(seq_def:list):
    '''
    given list of strings describing the sequential model
    :return: a sequential model
    '''
    layers = [build_layer(desc) for desc in seq_def]
    return nn.Sequential(*layers)


def build_layer(desc:str):
    '''
    given string description
    :return: corresponding layer
    '''
    tokens = desc.split()
    name = tokens[0]
    params = []
    for token in tokens[1:]:
        try:        p = int(token)
        except:     p = float(token)
        params.append(p)
    try:
        return getattr(nn, name)(*params)
    except:
        return getattr(custom_nn, name)(*params)


if __name__ == '__main__':
    filename = "configs/model.json"
    model = build_from_file(filename)
    print(model)