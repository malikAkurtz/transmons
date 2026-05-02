from Node import Node
from Branch import Branch

class Multidigraph():
    def __init__(self, nodes: list[Node], branches: list[Branch], s: list[int], t: list[int]):
        self.nodes = nodes
        self.branches = branches
        self.s = s
        self.t = t
