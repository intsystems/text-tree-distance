import zss
from .tree_format import Node, tree_with_context, json_to_node

def extract_sentences(tree: Node):
    '''
    A utility function that uses DFS to traverse a tree and extract all of its sentences.

    Arguments:
    tree - a text tree of the type Node.

    Output:
    sentences: list(string) - a list of all the sentences in tree.
    '''
    sentences = [Node.get_label(tree)]
    for child in Node.get_children(tree):
        sentences += extract_sentences(child)

    return sentences
    

def precompute_dists(tree_a: Node, tree_b: Node, encoder, embedding_dist):
    '''
    A helper function to precompute semantic distance between pairs of sentences in two text trees.

    Arguments:
    tree_a, tree_b - two trees of the zss type Node;
    similarity_func - a function of two string arguments which computes sentence distance.
    encoder - a language model's callable encoder that takes string input returns embeddings from the model
    embedding_dist - an embedding similarity measure that takes two embeddings as input and returns a non-negative distance value. 

    Output:
    sentence_dists: dict(dict(string: float)) - a 2-D dict containing scores for each pair of sentences from tree_a and tree_b.
    sentence_weights: dict(string: float) - a dict containing sentence weights (that is, distances to "") for deletion and insertion operation costs
    '''
    sentences_a, sentences_b = extract_sentences(tree_a), extract_sentences(tree_b)
    embeddings_a, embeddings_b = list(encoder(sentences_a)), list(encoder(sentences_b))

    sentence_dists = {}
    for sentence in sentences_a:
        sentence_dists[sentence] = {}

    for sent_a, emb_a in zip(sentences_a, embeddings_a):
        for sent_b, emb_b in zip(sentences_b, embeddings_b):
            sentence_dists[sent_a][sent_b] = embedding_dist(emb_a, emb_b)

    sentence_weights = {}
    empty_emb = encoder("")
    for sent, emb in zip(sentences_a + sentences_b, embeddings_a + embeddings_b):
        sentence_weights[sent] = embedding_dist(emb, empty_emb)

    return sentence_dists, sentence_weights


def text_tree_distance(tree_a: Node, tree_b: Node, encoder, embedding_dist, depth_factor=1.0, use_context=True):
    '''
    The function that calculates tree edit distance between to trees given a similarity function for sentence pairs.
    
    Arguments:
    tree_a, tree_b - two trees of the zss type Node to be compared;
    similarity_func - a function of two string arguments which computes sentence distance;
    depth_factor - a hyperparameter that scales sentence similarity based on the node's depth;
    use_context - a flag indicating whether parents of the given node will be used as context for sentence comparison.

    Output:
    dist: float - the calculated tree edit distance between tree_a and tree_b
    '''
    if use_context:
        tree_a = tree_with_context(tree_a)
        tree_b = tree_with_context(tree_b)

    sentence_dists, sentence_weights = precompute_dists(tree_a, tree_b, encoder, embedding_dist)

    # Here we define the update_cost, insert_cost and delete_cost functions needed for Zhang-Shasha's algorithm using the provided similarity function.
    def update_cost(node_a: Node, node_b: Node):
        return sentence_dists[Node.get_label(node_a)][Node.get_label(node_b)] * depth_factor**Node.get_depth(node_a)

    def insert_cost(node: Node):
        # We define node insertion and deletion costs as similarity of the node's label to an empty string
        return sentence_weights[Node.get_label(node)] * depth_factor**Node.get_depth(node)

    def remove_cost(node: Node):
        return sentence_weights[Node.get_label(node)] * depth_factor**Node.get_depth(node)

    dist = zss.distance(tree_a, tree_b, Node.get_children, insert_cost, remove_cost, update_cost)
    return dist


def text_tree_distance_w_o_precomputation(tree_a: Node, tree_b: Node, encoder, embedding_dist, depth_factor=1.0, use_context=True):
    '''
    An alternative version of text_tree_distance that doesn't incorporate precomputation.
    It calculates tree edit distance between to trees given a similarity function for sentence pairs.
    
    Arguments:
    tree_a, tree_b - two trees of the zss type Node to be compared;
    similarity_func - a function of two string arguments which computes sentence distance;
    depth_factor - a hyperparameter that scales sentence similarity based on the node's depth;
    use_context - a flag indicating whether parents of the given node will be used as context for sentence comparison.

    Output:
    dist: float - the calculated tree edit distance between tree_a and tree_b
    '''
    if use_context:
        tree_a = tree_with_context(tree_a)
        tree_b = tree_with_context(tree_b)

    def similarity_func(sentence_a, sentence_b):
        a_embedding = encoder(sentence_a)
        b_embedding = encoder(sentence_b)

        return embedding_dist(a_embedding, b_embedding)

    # Here we define the update_cost, insert_cost and delete_cost functions needed for Zhang-Shasha's algorithm using the provided similarity function.
    def update_cost(node_a: Node, node_b: Node):
        return similarity_func(Node.get_label(node_a), Node.get_label(node_b)) * depth_factor**Node.get_depth(node_a)

    def insert_cost(node: Node):
        # We define node insertion and deletion costs as similarity of the node's label to an empty string
        return similarity_func(Node.get_label(node), "")

    def remove_cost(node: Node):
        return similarity_func(Node.get_label(node), "")

    dist = zss.distance(tree_a, tree_b, Node.get_children, insert_cost, remove_cost, update_cost)
    return dist