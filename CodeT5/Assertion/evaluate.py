import editdistance
import re
from difflib import SequenceMatcher
from suffix_trees import STree
from tqdm import tqdm
import sys
from evaluation import Evaluation


gold = []
with open('model/assert/pl_ours/dev.gold') as f:
    for i in f:
        gold.append(i.split('\t')[1].strip())
pred = []
with open('model/assert/pl_ours/dev.output') as f:
    for i in f:
        pred.append(i.split('\t')[1].strip())

def calc_lcs(expected_orig, actual_orig):
    try:
        input_list = [expected_orig, actual_orig]
        st = STree.STree(input_list)
        longest_lcs = st.lcs()
    except RecursionError as e:
        print(e)
        print(f"error in calc_lcs for {self.expected_orig} and {self.actual_orig}")
        match = SequenceMatcher(None, expected_orig, actual_orig).find_longest_match(0, len(expected_orig), 0, len(actual_orig))
        longest_lcs = self.expected_orig[match.a:match.a + match.size]


    return longest_lcs

def edit_distance(expected_orig, actual_orig):
    return editdistance.eval(expected_orig, actual_orig)


is_match_count = 0
exact_match_count = 0
lcs = 0
ed = 0
for idx in tqdm(range(len(pred))):
    i = 'org . junit . Assert . '+pred[idx]
    j = 'org . junit . Assert . '+gold[idx]
    ed += edit_distance(i, j)
    lcs += len(calc_lcs(i,j))/len(j)
    evl = Evaluation(i, j)
    is_match = evl.is_match()
    if is_match:
        is_match_count += 1
    if i==j:
        exact_match_count+=1
print(round(exact_match_count*100/len(pred),2))
print(round(is_match_count*100/len(pred),2))
print(round(lcs*100/len(pred),2))
print(round(ed/len(pred),2))