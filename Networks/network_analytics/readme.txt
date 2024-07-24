networks_main: sets up and executes power.py, radial.py, and modularity.py (in that order).

modularity: Finds a weighted, louvain modularity for a network and supports networks_main. Can be executed alone.

power: Extracts degree distribution of network, supports networks_main.

radial: Finds radial distribution of network, supports networks_main.

random_network: Simulates a random network when executed (must input excel file).

simple_network: Graphs a network with label_propogation (not louvain) - faster visualization.