import networkx as nx

class GraphManager:
    def __init__(self):
        self.graph = nx.Graph()

    def add_or_update_edge(self, state, next_state, reward):
        edge_weight = -reward  # Using negative reward as cost
        if not self.graph.has_node(state):
            self.graph.add_node(state)
        if not self.graph.has_node(next_state):
            self.graph.add_node(next_state)
        self.graph.add_edge(state, next_state, weight=edge_weight)

    def shortest_path(self, source, target):
        return nx.shortest_path(self.graph, source=source, target=target, weight='weight')
