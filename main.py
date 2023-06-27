import gc

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from arguments_parser import arg_parser
from data import generate_dataloaders
from client import Client
from utils import *


if __name__ == "__main__":
    device = 'cuda'
    args = arg_parser()

    path = args.save_dir + args.name_experiment + '/'
    make_directory(path)

    client_models, global_models = generate_models(args)
    train_dataloaders, test_dataloaders, server_dataloader = generate_dataloaders(args)

    clients = []
    for idx in range(args.num_clients):
        clients.append(Client(idx, copy.deepcopy(client_models[idx]), args.lr, args.weight_decay, device,
                              train_dataloaders[idx], test_dataloaders[idx]))

    clustered = False
    loss_train, loss_test = [], []
    clustered_clients = [[] for cluster in range(args.nclusters)]
    for iteration in range(args.rounds):
        print(f'--- ROUND {iteration + 1} ---', flush=True)

        loss_clients = [0 for client in clients]
        count_data_clients = [0 for client in clients]
        for i in range(args.pi):
            for idx, client in enumerate(clients):
                loss, count = client.train_step()
                loss_clients[idx] += loss
                count_data_clients[idx] += count

            if clustered and i == (args.pi - 1):
                print('### Federated Aggregation', flush=True)
                for idx, cluster in enumerate(clustered_clients):
                    if len(cluster) > 0:
                        total_data = 0
                        for client_idx in cluster:
                            total_data += clients[client_idx].get_data_count('train')

                        weight_freq = []
                        for client_idx in cluster:
                            weight_freq.append(clients[client_idx].get_data_count('train') / total_data)

                        w_in_cluster = [copy.deepcopy(clients[client_idx].get_state_dict()) for client_idx in cluster]
                        global_models[idx] = copy.deepcopy(FedAvg(w_in_cluster, weight_freq))

                        for client_idx in cluster:
                            clients[client_idx].set_state_dict(global_models[idx])

        if (iteration + 1) % args.omega == 0:
            clustered = True
            print('### Clustering', flush=True)
            predictions = []
            for idx, client in enumerate(clients):
                predictions.append(client.inference(server_dataloader))
            predictions = np.array(predictions).squeeze(axis=-1)

            clustering_alg = AgglomerativeClustering(n_clusters=args.nclusters, linkage='ward')
            clustering_alg.fit(predictions)
            print(f'Clustering Result: {clustering_alg.labels_}', flush=True)

            for idx, client in enumerate(clients):
                clustered_clients[clustering_alg.labels_[idx]].append(idx)

        avg_train_loss = np.mean(
        np.array([loss_clients[idx] / count_data_clients[idx] for idx in range(len(clients))]))
        print(f'Averaged Train Loss: {avg_train_loss}', flush=True)
        loss_train.append(avg_train_loss)

        avg_test_loss = np.mean(np.array([client.test() for client in clients]))
        print(f'Averaged Test Loss: {avg_test_loss}', flush=True)
        loss_test.append(avg_test_loss)

        gc.collect()

    with open(path + 'loss_train.npy', 'wb') as fp:
        loss_train = np.array(loss_train)
        np.save(fp, loss_train)

    with open(path + 'loss_test.npy', 'wb') as fp:
        loss_train = np.array(loss_test)
        np.save(fp, loss_test)

    for idx, client in enumerate(clients):
        with open(path + f'client{idx}.pt', 'wb') as fp:
            torch.save(client.get_state_dict(), fp)
