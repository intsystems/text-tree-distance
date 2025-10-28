from .tree_format import TextTree
from edist.ted import ted
from edist.uted import uted
    

def precompute_dists(tree_a: TextTree, tree_b: TextTree, encoder, embedding_dist):
    '''
    A helper function to precompute semantic distance between pairs of sentences in two text trees.

    Arguments:
    tree_a, tree_b - two trees of the zss type TextTree;
    similarity_func - a function of two string arguments which computes sentence distance.
    encoder - a language model's callable encoder that takes string input and returns embeddings from the model
        Note that the encoder should be capable of processing arrays of strings.
    embedding_dist - an embedding similarity measure that takes two embeddings as input and returns a non-negative distance value. 

    Output:
    sentence_dists: dict(dict(string: float)) - a 2-D dict containing scores for each pair of sentences from tree_a and tree_b.
    sentence_weights: dict(string: float) - a dict containing sentence weights (that is, distances to "") for deletion and insertion operation costs
    '''
    sentences_a, sentences_b = tree_a.nodes, tree_b.nodes
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


def text_tree_distance(tree_1: TextTree, tree_2: TextTree, encoder, embedding_dist, unordered=True, use_context=False):
    '''
    The function that calculates tree edit distance between to trees given a similarity function for sentence pairs.
    
    Arguments:
    tree_a, tree_b - two trees of the zss type TextTree to be compared;
    similarity_func - a function of two string arguments which computes sentence distance;
    depth_factor - a hyperparameter that scales sentence similarity based on the node's depth;
    use_context - a flag indicating whether parents of the given node will be used as context for sentence comparison.

    Output:
    dist: float - the calculated tree edit distance between tree_a and tree_b
    '''
    tree_a = tree_1.copy()
    tree_b = tree_2.copy()
    
    if use_context:
        tree_a = tree_a.add_context()
        tree_b = tree_b.add_context()

    sentence_dists, sentence_weights = precompute_dists(tree_a, tree_b, encoder, embedding_dist)

    def update_cost(node_a: str, node_b: str):
        if node_a is None:
            return sentence_weights[node_b]
        if node_b is None:
            return sentence_weights[node_a]
        
        return sentence_dists[node_a][node_b]

    if unordered:
        dist = uted(*tree_a.nodes_and_adj(), *tree_b.nodes_and_adj(), update_cost)
    else:
        dist = ted(*tree_a.nodes_and_adj(), *tree_b.nodes_and_adj(), update_cost)
          
    return dist