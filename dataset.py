from itertools import permutations

import numpy as np

import tensorflow as tf
import tensorflow.keras as keras

from cyanide import *


def cos_angle(u, v):
    dot = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    return dot / norm_u / norm_v


def cif2key(cif):
    node_dict = {}
    edge_dict = {}
    with open(cif, "r") as f:
        for l in f:
            if not l.startswith("#"):
                continue

            tokens = l.split()
            if tokens[1] == "Topology":
                toponame = tokens[2][:-1]
            elif tokens[2] == "BuildingBlock:":
                t = tokens[1]
                bb = tokens[3][:-1]
                if "-" in t:
                    t = t.replace("[", "").replace("]", "")
                    t = [int(v) for v in t.replace("-", " ").split()]
                    t = tuple(t)
                    edge_dict[t] = bb
                else:
                    t = int(t.replace("[", "").replace("]", ""))
                    node_dict[t] = bb
            elif tokens[2] == "None":
                t = tokens[1]
                t = t.replace("[", "").replace("]", "")
                t = [int(v) for v in t.replace("-", " ").split()]
                t = tuple(t)
                edge_dict[t] = "None"

    key = toponame
    for bb in list(node_dict.values()) + list(edge_dict.values()):
        key += ":" + bb

    return key


class DataLoader:
    def __init__(self, topologies, node_hash, edge_hash, V):
        self.topologies = topologies
        self.node_hash = node_hash
        self.edge_hash = edge_hash
        self.V = V

    def key2data(self, key):
        n = 20
        m = 32
        beta = 0.1
        space = np.linspace(-1, 1, n)

        # Parse key information.
        tokens = key.split(":")
        toponame = tokens.pop(0)

        # Alias
        topologies = self.topologies
        node_hash = self.node_hash
        edge_hash = self.edge_hash
        V = self.V

        topo = topologies[toponame]

        node_dict = {}
        for t in topo.unique_node_types:
            node_dict[t] = tokens.pop(0)

        edge_dict = {}
        for i, j in topo.unique_edge_types:
            edge_dict[(i, j)] = tokens.pop(0)

        max_m = np.max(topo.unique_cn)
        max_n = topo.n_nodes
        N = np.zeros(max_n)
        NN = [[] for _ in range(max_n)]
        E = [[] for _ in range(max_n)]

        for i in topo.node_indices:
            nt = topo.get_node_type(i)
            nt = node_dict[nt]
            nt = node_hash[nt]
            N[i] = nt

        # Make local structure type.
        slot_type = {}
        for nt in topo.unique_node_types:
            pivot = np.argmax(topo.node_types == nt)

            Us = []
            for n1, n2 in permutations(topo.neighbor_list[pivot], 2):
                u = n1.distance_vector
                v = n2.distance_vector

                c = cos_angle(u, v)

                w = (c - space) / beta
                w = np.exp(-np.square(w))
                w /= np.sum(w)

                U = w @ V
                Us.append(U)
            Us = np.array(Us)
            U = np.mean(Us, axis=0)

            slot_type[nt] = U

        ST = []
        for nt in topo.node_types[topo.node_indices]:
            ST.append(slot_type[nt])
        ST = np.array(ST)

        for e in topo.edge_indices:
            ni, nj = topo.neighbor_list[e]
            i = ni.index
            j = nj.index

            t = tuple(topo.get_edge_type(e))
            t = edge_dict[t]
            t = edge_hash[t]
            E[j].append(t)
            E[i].append(t)

            NN[i].append(j)
            NN[j].append(i)

        for n in NN:
            while len(n) < max_m:
                n.append(-1)

        for n in E:
            while len(n) < max_m:
                n.append(-1)

        NN = np.array(NN)
        E = np.array(E)

        return N, NN, E, ST

    def cif2data(self, cif):
        key = cif2key(cif)
        return self.key2data(key)


def test():
    print("# TEST 1 #")
    cif = "../MOF_MAKER/examples/MOF-TEST/11727.cif"
    key = cif2key(cif)
    print(key)

    print("# TEST 2 #")
    topo_path = "/home/lsw/Workspace/MOF_MAKER/examples/topology.npz"
    topologies = {
        topo.name: topo for topo in np.load(topo_path)["topologies"]
    }
    node_hash = np.load("MOF-50000-sa-node_hash.npy").item()
    edge_hash = np.load("MOF-50000-sa-edge_hash.npy").item()
    print(edge_hash)
    V = np.load("MOF-50000-sa-V.npy")
    data_loader = DataLoader(topologies, node_hash, edge_hash, V)

    N, NN, E, ST = data_loader.cif2data(cif)
    print(N)
    print(NN)
    print(E)
    print(ST)

if __name__ == "__main__":
    test()