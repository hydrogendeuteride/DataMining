from collections import defaultdict


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

    if len(rowPtr) != n + 1:
        raise ValueError("rowPtr length must be n + 1")

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
        multiplied_pr = csr_multiplication(csr_matrix, pr)

        new_pr = [beta * x for x in multiplied_pr]
        s = sum(new_pr)
        new_pr = [x + (1 - s) / n for x in new_pr]

        error = max(abs(new_pr[i] - pr[i]) for i in range(n))
        if error < eps:
            break

        pr = new_pr

    return pr


edge_list= [
    (0, 5), (0, 8), (0, 3), (1, 7), (2, 8),
    (3, 6), (4, 5), (4, 2), (5, 1), (5, 2),
    (5, 7), (6, 0), (6, 3), (6, 7), (8, 7),
    (8, 0), (8, 2), (8, 3), (9, 7), (9, 3)
]

csr_matrix = graph_csr(edge_list)
csr_transposed = csr_transpose(csr_matrix)

pr = page_rank(csr_transposed, csr_transposed.n)

print("페이지 랭크 결과:")
for index, score in enumerate(pr):
    node_id = csr_matrix.index_to_node[index]
    print(f"노드 {node_id}: {score}")

print("=================================")
g = process_file("web-Google.txt")
csr_matrix_large = graph_csr(g)
csr_transposed_large = csr_transpose(csr_matrix_large)

print(f"노드 수: {csr_matrix_large.n}, 엣지 수: {len(csr_matrix_large.value)}")
print("=================================")
pr_large = page_rank(csr_transposed_large, csr_transposed_large.n)

s_t = sorted(enumerate(pr_large), key=lambda x: x[1], reverse=True)
top_n = 10
top_n_values = [value for index, value in s_t[:top_n]]
top_n_indices = [index for index, value in s_t[:top_n]]
top_n_node_ids = [csr_matrix_large.index_to_node[index] for index in top_n_indices]

print("상위 노드들의 페이지 랭크 값:")
for node_id, value in zip(top_n_node_ids, top_n_values):
    print(f"노드 {node_id}: {value}")
