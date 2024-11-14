import csv
from itertools import combinations
import time

class Apriori:
    def __init__(self, file, min_support=0.02, min_confidence=0.05):
        self.file = file
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.transaction_count = self.get_transaction_count()

    def get_transaction_count(self):
        with open(self.file, newline='') as csvFile:
            reader = csv.reader(csvFile)
            return sum(1 for _ in reader)

    def support(self):
        item_counts = {}
        with open(self.file, newline='') as csvFile:
            reader = csv.reader(csvFile)
            for row in reader:
                for item in row:
                    if item:
                        item_counts[item] = item_counts.get(item, 0) + 1
        frequent_items = {(item,): count for item, count in item_counts.items()
                          if count / self.transaction_count >= self.min_support}
        return frequent_items

    def k_support(self, cur_itemset):
        itemsets = {}
        with open(self.file, newline='') as csvFile:
            reader = csv.reader(csvFile)
            for row in reader:
                row_set = set(row)
                for item in cur_itemset:
                    if set(item).issubset(row_set):
                        itemsets[item] = itemsets.get(item, 0) + 1
        return {itemset: count for itemset, count in itemsets.items()
                if count / self.transaction_count >= self.min_support}

    def possible_candidates(self, prev_freq_items, k):
        prev_freq_items_list = list(prev_freq_items.keys())
        candidates = []
        len_prev = len(prev_freq_items_list)
        for i in range(len_prev):
            for j in range(i + 1, len_prev):
                l1 = list(prev_freq_items_list[i])
                l2 = list(prev_freq_items_list[j])
                l1.sort()
                l2.sort()
                if l1[:k - 2] == l2[:k - 2]:
                    candidate = tuple(sorted(set(prev_freq_items_list[i]) | set(prev_freq_items_list[j])))
                    if len(candidate) == k:
                        candidates.append(candidate)
        return candidates

    def prune_candidates(self, candidates, prev_freq_items):
        pruned_candidates = []
        prev_freq_items_set = set(prev_freq_items.keys())

        for candidate in candidates:
            all_subsets_freq = True
            subsets = combinations(candidate, len(candidate) - 1)
            for subset in subsets:
                if subset not in prev_freq_items_set:
                    all_subsets_freq = False
                    break
            if all_subsets_freq:
                pruned_candidates.append(candidate)
        return pruned_candidates

    def frequent_itemsets(self):
        freq_items = self.support()

        k = 2
        all_freq_itemsets = [freq_items]

        while freq_items:
            candidates = self.possible_candidates(freq_items, k)
            candidates = self.prune_candidates(candidates, freq_items)

            freq_items = self.k_support(candidates)

            if freq_items:
                all_freq_itemsets.append(freq_items)

            k += 1

        return all_freq_itemsets

    def get_support_count(self, freq_itemsets, itemset):
        for k_itemsets in freq_itemsets:
            if itemset in k_itemsets:
                return k_itemsets[itemset]
        return 0

    def association_rules(self):
        rules = []
        freq_itemsets = self.frequent_itemsets()

        for k_itemsets in freq_itemsets[1:]:
            for itemset, support in k_itemsets.items():
                itemset_size = len(itemset)

                for i in range(1, itemset_size):
                    for antecedent in combinations(itemset, i):
                        antecedent = tuple(sorted(antecedent))
                        consequent = tuple(sorted(set(itemset) - set(antecedent)))

                        if consequent:
                            antecedent_support_count = self.get_support_count(freq_itemsets, antecedent)
                            confidence = support / antecedent_support_count

                            if confidence >= self.min_confidence:
                                consequent_support_count = self.get_support_count(freq_itemsets, consequent)
                                consequent_support_ratio = consequent_support_count / self.transaction_count

                                interest = abs(confidence - consequent_support_ratio)

                                rule = {
                                    'antecedent': antecedent,
                                    'consequent': consequent,
                                    'support': support / self.transaction_count,
                                    'confidence': confidence,
                                    'interest': interest
                                }
                                rules.append(rule)

        return rules


if __name__ == '__main__':
    file_name = 'market.csv'

    start_time = time.time()
    apriori = Apriori(file_name, min_support=0.02, min_confidence=0.3)

    frequent_itemsets = apriori.frequent_itemsets()
    print(frequent_itemsets)

    rules = apriori.association_rules()
    end_time = time.time()

    print(f"Association Rules: {len(rules)}")

    min_interest, max_interest = 0.99, -0.99

    for rule in rules:
        antecedent = ", ".join(rule['antecedent'])
        consequent = ", ".join(rule['consequent'])
        print(f"Rule: {antecedent} -> {consequent}, Support: {rule['support']:.4f}, "
              f"Confidence: {rule['confidence']:.4f}, Interest: {rule['interest']:.4f}")

        min_interest = min(min_interest, rule['interest'])
        max_interest = max(max_interest, rule['interest'])

    print(f"min interest: {min_interest}, max interest:  {max_interest}")

    execution_time = end_time - start_time
    print(f"execution time: {execution_time:.4f}sec")