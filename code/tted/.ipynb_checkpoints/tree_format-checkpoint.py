import json
from edist import tree_utils

class TextTree:
    '''
    A class for text tree objects we'll be working with.

    A tree in the edist library is defined by:
    1) a node list (a list of strings in our case);
    2) an adjacency list (a list of lists of integers representing each node's child node indices).
    '''
    def __init__(self, nodes, adj):
        tree_utils.check_tree_structure(adj)
        self.nodes = nodes
        self.adj = adj
        self.depths = TextTree.get_depths(nodes, adj)

    @staticmethod
    def get_depths(nodes, adj):
        depths = {0: 1}
        for i in range(len(nodes)):
            for child in adj[i]:
                depths[child] = depths[i] + 1

        return depths

    def nodes_and_adj(self):
        return (self.nodes, self.adj)

    def copy(self):
        return TextTree(self.nodes.copy(), [a.copy() for a in self.adj])

    def __str__(self):
        string = tree_utils.tree_to_string(self.nodes, self.adj, indent=True)
        return string

    def __len__(self):
        return len(self.nodes)

    def at(self, depth: int):
        '''
        Returns new TextTree that is a copy of self trimmed to the specified depth.
        '''
        if depth < 1:
            raise TypeError("The depth parameter must be a positive integer.")
        
        new_nodes = [self.nodes[i] for i in range(len(self)) if self.depths[i] <= depth]        
        new_adj = [self.adj[i] if self.depths[i] < depth else [] for i in range(len(self)) if self.depths[i] <= depth]
        
        selected_nodes = [i for i in range(len(self)) if self.depths[i] <= depth]
        new_node_mapping = {selected_nodes[i]: i for i in range(len(selected_nodes))}
        for i in range(len(new_adj)):
            new_adj[i] = [new_node_mapping[node] for node in new_adj[i]]

        return TextTree(new_nodes, new_adj)

    def add_context(self):
        '''
        A function that creates a relabeled tree based on the input one by adding sentences from parent nodes as context to child nodes.
        Returns new relabeled TextTree object.
        '''
        def _add_context_to_node(node_no, context):
            new_node = context + ' ' + self.nodes[node_no]
            new_tree.nodes[node_no] = new_node
            
            for child_node_no in self.adj[node_no]:
                _add_context_to_node(child_node_no, new_node)
            
        new_tree = self.copy()
        _add_context_to_node(0, "")
        
        return new_tree
        

    @staticmethod
    def from_json(filename):
        '''
        Read text tree from JSON file and convert it to a TextTree object.
    
        Arguments:
        filename: string - name of JSON file with text tree to open.
    
        Output:
        tree: TextTree - text tree object.
        '''
        with open(filename, 'r', encoding="utf8") as f:
            json_data = json.load(f)

        nodes, adj = TextTree.dict_to_nodes_and_adj(json_data)

        tree = TextTree(nodes, adj)
        return tree

    @staticmethod
    def dict_to_nodes_and_adj(json_data, node_no=0):
        '''
        Helper function that constructs a tree in node list + adjacency list format from a dict recursively using DFS.
    
        Arguments:
        json_data: dict - tree in dict format.
    
        Output:
        nodes: list[string] - list of tree nodes.
        adj: list[list[int]] - adjacency list.
        '''
        if not json_data:
            return None
        
        # The JSON format has exactly one key-value pair per subtree
        label, children_dict = next(iter(json_data.items()))
    
        nodes = [label]
        adj = [[]]
    
        for child_label, child_subtree in children_dict.items():
            child_node_no = node_no + len(nodes) # This is the next unused node index that we assign to the new child node.
            adj[0].append(child_node_no)
            child_nodes, child_adj = TextTree.dict_to_nodes_and_adj({child_label: child_subtree}, child_node_no)
            
            if child_nodes:
                nodes.extend(child_nodes)
                adj.extend(child_adj)
        
        return nodes, adj