from collections import defaultdict
import argparse


class CSRMatrix:
    def __init__(self, value, index, rowptr, n, m, node_to_index=None, index_to_node=None):
        self.value = value
        self.index = index
        self.rowptr = rowptr
        self.n = n
        self.m = m
        self.node_to_index = node_to_index
        self.index_to_node = index_to_node


def process_file(filename):
    graph = []
    with open(filename, newline='') as file:
        for lines in file:
            if lines.startswith('#'):
                continue

            src, dst = lines.strip().split('\t')
            graph.append((int(src), int(dst)))

    return graph


def graph_csr(edge_list):
    value = []
    index = []
    rowptr = [0]

    graph = defaultdict(list)
    out_degree = defaultdict(int)
    all_nodes = set()

    for src, dst in edge_list:
        graph[src].append(dst)
        out_degree[src] += 1
        all_nodes.add(src)
        all_nodes.add(dst)

    nodes = sorted(all_nodes)
    node_to_index = {node: i for i, node in enumerate(nodes)}
    index_to_node = {i: node for node, i in node_to_index.items()}

    n, m = len(nodes), len(nodes)

    for node in nodes:
        for dst in graph[node]:
            value.append(1 / out_degree[node])
            index.append(node_to_index[dst])
        rowptr.append(len(value))

    return CSRMatrix(value, index, rowptr, n, m, node_to_index, index_to_node)


def csr_transpose(csr_matrix):
    value = csr_matrix.value
    colIndex = csr_matrix.index
    rowPtr = csr_matrix.rowptr
    n = csr_matrix.n
    m = csr_matrix.m

    nz = len(value)

    res_value = [0.0] * nz
    res_colIndex = [0] * nz
    res_rowPtr = [0] * (m + 1)

    for col in colIndex:
        res_rowPtr[col + 1] += 1

    for i in range(2, m + 1):
        res_rowPtr[i] += res_rowPtr[i - 1]

    for i in range(n):
        for j in range(rowPtr[i], rowPtr[i + 1]):
            col = colIndex[j]
            dest = res_rowPtr[col]

            res_value[dest] = value[j]
            res_colIndex[dest] = i

            res_rowPtr[col] += 1

    for i in range(m, 0, -1):
        res_rowPtr[i] = res_rowPtr[i - 1]
    res_rowPtr[0] = 0

    res_n = m
    res_m = n

    return CSRMatrix(res_value, res_colIndex, res_rowPtr, res_n, res_m,
                     csr_matrix.node_to_index, csr_matrix.index_to_node)


def csr_multiplication(csr_matrix, x):
    value = csr_matrix.value
    index = csr_matrix.index
    rowptr = csr_matrix.rowptr
    n = csr_matrix.n

    y = [0] * n

    for i in range(n):
        row_start = rowptr[i]
        row_end = rowptr[i + 1]

        for j in range(row_start, row_end):
            y[i] += value[j] * x[index[j]]

    return y


def page_rank(csr_matrix, n, iter=64, beta=0.85, eps=1e-8):
    pr = [1 / n] * n
    new_pr = [0.0] * n

    for t in range(0, iter):
        print(f"Iteration {t+1}")
        multiplied_pr = csr_multiplication(csr_matrix, pr)

        new_pr = [beta * x for x in multiplied_pr]
        s = sum(new_pr)
        new_pr = [x + (1 - s) / n for x in new_pr]

        error = max(abs(new_pr[i] - pr[i]) for i in range(n))
        if error < eps:
            break

        pr = new_pr

    print("converged at iteration {}".format(t+1))
    return pr


def main(args):
    G = process_file(args.input)
    csr_matrix_large = graph_csr(G)
    csr_transposed_large = csr_transpose(csr_matrix_large)

    pr_large = page_rank(csr_transposed_large, csr_transposed_large.n, iter=args.niter)

    s_t = sorted(enumerate(pr_large), key=lambda x: x[1], reverse=True)

    value = [value for index, value in s_t]
    index = [index for index, value in s_t]
    ids = [csr_matrix_large.index_to_node[index] for index in index]

    f = open(args.output, "w")
    for node_id, value in zip(ids, value):
        f.write(f"{node_id}\t{value:.6f}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", action="store")
    parser.add_argument("--output", dest="output", action="store")
    parser.add_argument("--niter", dest="niter", default=64, action="store")
    args = parser.parse_args()

    main(args)
