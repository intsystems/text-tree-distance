import zss
import json

class Node(zss.Node):
    '''
    Override of the zss.Node class containing a node depth parameter.
    It can be used to adjust editing costs depending on node depth.
    '''
    def __init__(self, label, children=None, depth=0):
        '''
        Updated version of Node constructor containing self.depth initialization.
        '''
        self.label = label
        self.children = children or list()
        self.depth = depth

    @staticmethod
    def get_depth(node):
        '''
        Method that returns the node's depth field.
        '''
        return node.depth
    
    def __str__(self):
        string = '-' * self.depth + self.label + '\n'

        for child in Node.get_children(self):
            string += str(child)

        return string


def dict_to_node(json_data: dict, depth=0):
    '''
    Recursively construct tree from dict data.
    '''
    if not json_data:
        return None

    # The JSON format has exactly one key-value pair per subtree
    label, children_dict = next(iter(json_data.items()))

    node = Node(label, depth=depth)

    for child_label, child_subtree in children_dict.items():
        child_node = dict_to_node({child_label: child_subtree}, depth=depth+1)
        if child_node:
            node.addkid(child_node)

    return node

def json_to_node(filename):
    '''
    Reads text tree from JSON file to tree of the Node class.

    Arguments:
    filename: string - name of JSON file with text tree to open.

    Output:
    tree: Node - root node of the tree.
    '''
    with open(filename, 'r') as f:
        json_data = json.load(f)

    tree = dict_to_node(json_data)
    return tree
    

def tree_with_context(input_node: Node):
    '''
    A function that creates a relabeled tree based on the input one by adding sentences from parent nodes as context to child nodes.

    Arguments:
    input_node - root node of the tree to be relabeled.

    Output: 
    Root node of relabeled tree.
    '''
    def add_context(node: Node, context):
        new_node = Node(context + Node.get_label(node), depth=Node.get_depth(node))
        
        for child_node in Node.get_children(node):
            new_node.addkid(add_context(child_node, Node.get_label(new_node)))
        
        return new_node

    return add_context(input_node, "")