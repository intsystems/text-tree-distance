from .tree_format import TextTree
from edist.ted import ted
from edist.uted import uted
    

def precompute_dists(tree_a: TextTree, tree_b: TextTree, encoder, embedding_dist):
    '''
    A helper function to precompute semantic distance between pairs of sentences in two text trees.

    Arguments:
    tree_a, tree_b: TextTree - two text tree instances;
    encoder - function that encodes text to vectors;
    embedding_dist - function used to measure distance between text embeddings;

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


def tted(
    tree_1: TextTree, 
    tree_2: TextTree, 
    encoder, 
    embedding_dist, 
    normalize: bool = False, 
    unordered: bool = True, 
    use_context: bool = False, 
    at: int | None = None
):
    '''
    The function that calculates tree edit distance between to trees given a similarity function for sentence pairs.
    
    Arguments:
    tree_1, tree_2: TextTree - two text tree instances to be compared;
    encoder - function that encodes text to vectors;
    embedding_dist - function used to measure distance between text embeddings;
    normalize: bool - flag indicating whether the distance is normalized;
    unordered: bool - flag indicating whether the trees are considered unordered;
    use_context: bool - a flag indicating whether parents of the given node will be used as context for sentence comparison;
    at: int | None - If not None, the trees are trimmed to the specified depth (TTED@k).

    Output:
    dist: float - the calculated tree edit distance between tree_a and tree_b
    '''
    tree_a = tree_1.copy()
    tree_b = tree_2.copy()

    if at is not None:
        tree_a = tree_a.at(at)
        tree_b = tree_b.at(at)
    
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

    if normalize:
        dist = 2 * dist / (float(max(sentence_weights.values())) * (len(tree_a) + len(tree_b)) + dist)
          
    return dist

def avg_tted(
    tree_1: TextTree, 
    tree_2: TextTree, 
    encoder, 
    embedding_dist, 
    unordered: bool = True, 
    use_context: bool = False,
    at: int | None = None,
):
    '''
    Function that calculates AvgTTED - average normalized TTED@k for all depths k (or up to a certain depth if specified).
    
    Arguments:
    tree_1, tree_2: TextTree - two text tree instances to be compared;
    encoder - function that encodes text to vectors;
    embedding_dist - function used to measure distance between text embeddings;
    unordered: bool - flag indicating whether the trees are considered unordered;
    use_context: bool - a flag indicating whether parents of the given node will be used as context for sentence comparison;
    at: int | None - If not None, AvgTTED@k will be computed (up to the specified depth).

    Output:
    dist: float - the calculated tree edit distance between tree_1 and tree_2
    '''
    max_depth = max((max(tree_1.depths.values()), max(tree_2.depths.values())))
    if at is not None:
        max_depth = min(at, max_depth)

    dists = []
    for depth in range(1, max_depth+1):
        dists.append(
            tted(tree_1, tree_2, encoder, embedding_dist, normalize=True, unordered=unordered, use_context=use_context, at=depth)
        )

    return sum(dists) / len(dists)