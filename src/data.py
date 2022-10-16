import numpy as np
import torch
import networkx as nx


def load_coil_graphs(root="data/"):
    with open(f"{root}COIL-RAG/COIL-RAG_graph_labels.txt") as f:
        labels = list(map(lambda x: int(x) - 1, f.readlines()))

    with open(f"{root}COIL-RAG/COIL-RAG_graph_indicator.txt") as f:
        indicators = list(map(lambda x: int(x) - 1, f.readlines()))

    with open(f"{root}COIL-RAG/COIL-RAG_A.txt") as f:
        edges = list(map(lambda x: (int(x.split(",")[0]) - 1, int(x.split(",")[1]) - 1), f.readlines()))

    with open(f"{root}COIL-RAG/COIL-RAG_node_attributes.txt") as f:
        node_attrs = list(map(lambda x: np.array(list(map(float, x.split(", ")))), f.readlines()))

    with open(f"{root}COIL-RAG/COIL-RAG_edge_attributes.txt") as f:
        edge_attrs = list(map(float, f.readlines()))

    graphs = [nx.Graph() for i in range(len(labels))]

    for i, (feat, g) in enumerate(zip(node_attrs, indicators)):
        graphs[g].add_node(i, hist=feat)

    for i, (length, (n1, n2)) in enumerate(zip(edge_attrs, edges)):
        g1, g2 = indicators[n1], indicators[n2]
        assert g1 == g2

        graphs[g1].add_edge(n1, n2, length=length)

    return graphs, labels


class CoilDataset(torch.utils.data.Dataset):
    def __init__(self, coil_graphs, targets):
        self.coil_graphs = coil_graphs
        self.targets = targets

    def __len__(self):
        return len(self.coil_graphs)

    def __getitem__(self, idx):
        graph = self.coil_graphs[idx]
        node_features = torch.vstack([torch.tensor(graph.nodes[i]["hist"]) for i in graph.nodes])
        node_centralities = [graph.degree[i] for i in graph.nodes]

        shortest_paths_dict = dict(nx.all_pairs_shortest_path(graph))
        paths = [[shortest_paths_dict[v][u] for u in graph.nodes] for v in graph.nodes]
        shortest_paths_lengths_dict = dict(nx.all_pairs_shortest_path_length(graph))
        path_lengths = [[shortest_paths_lengths_dict[v][u] for u in graph.nodes] for v in graph.nodes]

        for u in range(len(graph.nodes)):
            for v in range(len(graph.nodes)):
                p = paths[u][v]
                features = torch.zeros(len(p) - 1, 1)
                for i in range(len(p) - 1):
                    features[i] = graph.get_edge_data(p[i], p[i+1])["length"]
                paths[u][v] = features

        return {"node_features": node_features, "target": self.targets[idx], 
                "node_centralities": node_centralities, "paths": paths, "path_lengths": path_lengths}


def collate_graphs(data):
    max_nodes = max([x["node_features"].shape[0] for x in data])
    mask = torch.zeros(len(data), max_nodes)
    node_features = torch.zeros(len(data), max_nodes, data[0]["node_features"].shape[1])
    node_centralities = torch.zeros(len(data), max_nodes)

    for i, x in enumerate(data):
        mask[i, :x["node_features"].shape[0]] = 1
        node_features[i, :x["node_features"].shape[0]] = x["node_features"]
        node_centralities[i, :x["node_features"].shape[0]] = torch.tensor(x["node_centralities"])

    target = torch.tensor([x["target"] for x in data])
    paths = [x["paths"] for x in data]

    path_lengths = torch.zeros((len(data), max_nodes, max_nodes))

    for i, x in enumerate(data):
        path_lengths[i, :x["node_features"].shape[0], :x["node_features"].shape[0]] = torch.tensor(x["path_lengths"])
    
    return {"node_features": node_features, "mask": mask, "target": target,
            "node_centralities": node_centralities, "paths": paths, "path_lengths": path_lengths}
