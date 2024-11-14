from efficient_apriori import apriori
import csv
from Apriori import Apriori


def load_transactions(file):
    with open(file, newline='') as csvFile:
        reader = csv.reader(csvFile)
        transactions = [tuple(row) for row in reader if row]
    return transactions


def test_efficient_apriori(file, min_support, min_confidence):
    transactions = load_transactions(file)
    itemsets, rules = apriori(transactions, min_support=min_support, min_confidence=min_confidence)
    return itemsets, rules


def test_custom_apriori(file, min_support, min_confidence):
    apriori_instance = Apriori(file, min_support=min_support, min_confidence=min_confidence)
    frequent_itemsets = apriori_instance.frequent_itemsets()
    rules = apriori_instance.association_rules()
    return frequent_itemsets, rules


def compare_itemsets(eff_itemsets, custom_itemsets):
    eff_itemsets_flat = {frozenset(itemset) for level in eff_itemsets.values() for itemset in level}
    custom_itemsets_flat = {frozenset(itemset) for level in custom_itemsets for itemset in level.keys()}

    return eff_itemsets_flat == custom_itemsets_flat


def compare_rules(custom_rules, apriori_rules):
    custom_rules_flat = []
    for rule in custom_rules:
        custom_rules_flat.append((set(rule['antecedent']), set(rule['consequent']), rule['support'], rule['confidence']))

    apriori_rules_flat = []
    for rule in apriori_rules:
        apriori_rules_flat.append((set(rule.lhs), set(rule.rhs), rule.support, rule.confidence))

    custom_rules_flat.sort(key=lambda x: (len(x[0]), len(x[1]), sorted(x[0]), sorted(x[1])))
    apriori_rules_flat.sort(key=lambda x: (len(x[0]), len(x[1]), sorted(x[0]), sorted(x[1])))

    if len(custom_rules_flat) != len(apriori_rules_flat):
        print(f"different rule number: {len(custom_rules_flat)} vs {len(apriori_rules_flat)}")

    for i, (custom_lhs, custom_rhs, custom_support, custom_confidence) in enumerate(custom_rules_flat):
        apriori_lhs, apriori_rhs, apriori_support, apriori_confidence = apriori_rules_flat[i]
        if custom_lhs != apriori_lhs or custom_rhs != apriori_rhs:
            print(f"rule mismatch {i}: {custom_lhs} => {custom_rhs} vs {apriori_lhs} => {apriori_rhs}")
        elif abs(custom_support - apriori_support) > 1e-6:
            print(f"support mismatch {custom_lhs} => {custom_rhs}: {custom_support} vs {apriori_support}")
        elif abs(custom_confidence - apriori_confidence) > 1e-2:
            print(
                f"confidence mismatch {custom_lhs} => {custom_rhs}: {custom_confidence} vs {apriori_confidence}")
        else:
            print(f"rule match {custom_lhs} => {custom_rhs} support: {custom_support:.2f} confidence: {custom_confidence:.2f}")


def compare_results(file, min_support, min_confidence):
    eff_itemsets, eff_rules = test_efficient_apriori(file, min_support, min_confidence)

    custom_itemsets, custom_rules = test_custom_apriori(file, min_support, min_confidence)

    itemsets_match = compare_itemsets(eff_itemsets, custom_itemsets)
    print("itemsets match:", itemsets_match)

    compare_rules(custom_rules, eff_rules)


file = 'market.csv'
min_support = 0.02
min_confidence = 0.03

compare_results(file, min_support, min_confidence)
