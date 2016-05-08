run -np 4 --hostfile myhostfile  --mca btl_tcp_if_exclude 172.0.0.0/8 --map-by node ./BFS -g ../../../Tables/swarm_swarm_similarity_ds0409.mtx -gtype 1
