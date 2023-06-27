import argparse


def arg_parser():
    parser = argparse.ArgumentParser()

    # federated arguments
    parser.add_argument('--num_clients', type=int, default=5, help="number of clients")
    parser.add_argument('--rounds', type=int, default=200, help="federated rounds")
    parser.add_argument('--omega', type=int, default=5, help="federated aggregation pace")
    parser.add_argument('--pi', type=int, default=25, help="training pace per federated iteration")
    parser.add_argument('--nclusters', type=int, default=3, help="number of clusters")

    # model arguments
    parser.add_argument('--model', type=str, default='resnet-18', help='model name')
    parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
    parser.add_argument('--batch_size', type=int, default=4, help="batch size")
    parser.add_argument('--device', type=str, default='cuda', help="device")
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')

    # dataset argument
    parser.add_argument('--data_dir', type=str, default='../data/', help='data directory of data distributions')

    # loging arguments
    parser.add_argument('--name_experiment', type=str, default='sample', help='name of the experiment')
    parser.add_argument('--save_dir', type=str, default='../save_results/', help='save directory')

    args = parser.parse_args()
    return args
