import copy
import os

from models import *


def make_directory(path):
    try:
        os.makedirs(path)
    except Exception as _:
        pass


def generate_models(args):
    client_models, global_models = [], []
    for client in range(args.num_clients + 1):
        if args.model == 'resnet-18':
            model = generate_resnet(18).to(args.device)
        else:
            print("model not supported")
            exit(1)

        if client == 0:
            temp_model = copy.deepcopy(model)
            for _ in range(args.nclusters):
                global_models.append(copy.deepcopy(temp_model.state_dict()))
        else:
            client_models.append(copy.deepcopy(model))

    return client_models, global_models


def FedAvg(w, weight_avg=None):
    if weight_avg is None:
        weight_avg = [1 / len(w) for _ in range(len(w))]

    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = w_avg[k].cuda() * weight_avg[0]

    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] = w_avg[k].cuda() + w[i][k].cuda() * weight_avg[i]
    return w_avg
