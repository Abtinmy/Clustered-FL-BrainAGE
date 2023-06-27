python ../main.py \
--name_experiment='sample' \
--rounds=200 \
--num_clients=5 \
--lr=0.0001 \
--model='resnet-18' \
--pi=50 \
--omega=5 \
--ncluster=3 \
--data_dir='../../data/' \
--save_dir='../save_results/' \
2>&1 | tee '../logs.txt'