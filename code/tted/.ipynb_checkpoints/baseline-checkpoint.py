from .tree_format import Node
import rouge
import math

'''
Here we define functions that implement the mind map generation scoring method from the 2024 paper
"Coreference Graph Guidance for Mind-Map Generation" by Zhang et al. 
Note that, while trying to stay true to the original realization of the method, we had to modify 
the code slightly to get it working and added a wrapper to make it compatible with our tree format.

The full original code from the paper mentioned can be found at https://github.com/Cyno2232/CMGN
'''

def rouge_sim2(summary, hypothesis):
    '''
    The function that calculates averaged ROUGE similarity between two sentences.
    The similarity score is defined as the mean of ROUGE-1, ROUGE-2 and ROUGE-L, as in Zhang et al., 2024.

    Arguments:
    summary, hypothesis: string - two sentences to be compared.

    Output:
    score: float - sentence similarity score.
    '''
    if summary == []:
        summary = ""
    if hypothesis == []:
        hypothesis = ""
    evaluator = rouge.Rouge()
    r_1 = evaluator.get_scores(hypothesis, summary)[0]['rouge-1']['f']
    r_2 = evaluator.get_scores(hypothesis, summary)[0]['rouge-2']['f']
    r_l = evaluator.get_scores(hypothesis, summary)[0]['rouge-l']['f']
    score = r_1 / 3 + r_2 / 3 + r_l / 3
    return score


def compare_method(pairs1, pairs2):
    '''
    A function that compares two text trees using the above similarity function.

    Arguments:
    pairs, pairs2: list[list[string]] - two text trees in the form of [parent, child] sentence pairs in width-first order.

    Output:
    sim: float - similarity score for the two trees. 
    '''
    pairs = pairs1.copy()
    sim = 0
    for i in range(len(pairs2)):
        msp = ['_', '_']
        found = False
        for j in range(len(pairs)):
            first = rouge_sim2(pairs2[i][0], msp[0]) + rouge_sim2(pairs2[i][1], msp[1])
            second = rouge_sim2(pairs2[i][0], pairs[j][0]) + rouge_sim2(pairs2[i][1], pairs[j][1])
            if first < second:
                msp = pairs[j]
                max_index = j
                found = True

        cur_sim_0 = rouge_sim2(pairs2[i][0], msp[0])
        cur_sim_1 = rouge_sim2(pairs2[i][1], msp[1])
        sim = sim + (cur_sim_0 + cur_sim_1) / 2

        if found:
            del pairs[max_index]
    return sim


def tree_to_pairs(tree: Node):
    '''
    Function that transforms Node tree to array of parent-child pairs.

    Arguments:
    tree: Node - input tree of the Node class.

    Output:
    pairs: list[list[string]] - list of pairs of parent-child node labels in width-first order.
    '''
    pairs = []
    for child in Node.get_children(tree):
        pairs.append([Node.get_label(tree), Node.get_label(child)])
    for child in Node.get_children(tree):
        pairs += tree_to_pairs(child)

    return pairs


def baseline_similarity(tree_a: Node, tree_b: Node):
    '''
    Function that scores tree similarity using the baseline method. 

    Arguments:
    tree_a, tree_b: Node - trees to be compared.

    Output:
    dist: float - similarity score between the input trees.
    '''
    pairs_a, pairs_b = tree_to_pairs(tree_a), tree_to_pairs(tree_b)

    sim = compare_method(pairs_a, pairs_b)

    return sim


def baseline_distance(tree_a: Node, tree_b: Node):
    '''
    Function that scores tree distance using the baseline method. 
    To make a distance metric out of a similarity measure, we convert it into a pseudometric using the kernel method.

    Arguments:
    tree_a, tree_b: Node - trees to be compared.

    Output:
    dist: float - distance score between the input trees.
    '''
    sim = baseline_similarity(tree_a, tree_b) + baseline_similarity(tree_b, tree_a)
    max_sim = baseline_similarity(tree_a, tree_a) + baseline_similarity(tree_b, tree_b)
    
    dist = math.sqrt(max_sim - sim)

    return dist