from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix
import networkx as nx


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
    all_nodes = set()

    for src, dst in edge_list:
        graph[src].append(dst)
        all_nodes.add(src)
        all_nodes.add(dst)

    nodes = sorted(all_nodes)
    node_to_index = {node: i for i, node in enumerate(nodes)}

    n, m = max(nodes) + 1, max(nodes) + 1

    for node in nodes:
        for dst in graph[node]:
            value.append(1)
            index.append(node_to_index[dst])
        rowptr.append(len(value))

    return value, index, rowptr, n, m

edge_list = [
    (0, 5), (0, 8), (0, 3), (1, 7), (2, 8),
    (3, 6), (4, 5), (4, 2), (5, 1), (5, 2),
    (5, 7), (6, 0), (6, 3), (6, 7), (8, 7),
    (8, 0), (8, 2), (8, 3), (9, 7), (9, 3)
]

# edge_list_2 = [
#     (0, 0), (0, 2), (1, 0), (1, 1), (2, 1), (2, 2)
# ]

value, index, rowptr, n, m = graph_csr(edge_list)

print(value, len(value))
print(index, len(index))
print(rowptr, len(rowptr))

csr_mtx = csr_matrix((value, index, rowptr), shape=(n, m))

dense_matrix = csr_mtx.toarray()
print(dense_matrix)


def csr_transpose(csr_matrix):
    value, colIndex, rowPtr, n, m = csr_matrix
    print(n, m)
    nz = len(value)

    res_value = [0.0] * nz
    res_colIndex = [0] * nz
    res_rowPtr = [0] * (m + 1)

    for col in colIndex:
        res_rowPtr[col + 1] += 1

    for i in range(2, m + 1):
        res_rowPtr[i] += res_rowPtr[i - 1]

    print(res_rowPtr)

    for i in range(n):
        for j in range(rowPtr[i], rowPtr[i + 1]):
            col = colIndex[j]
            dest = res_rowPtr[col]

            res_value[dest] = value[j]
            res_colIndex[dest] = i

            res_rowPtr[col] += 1

    print(res_rowPtr)
    for i in range(m, 0, -1):
        res_rowPtr[i] = res_rowPtr[i - 1]
    res_rowPtr[0] = 0
    print(res_rowPtr)

    res_n = m
    res_m = n

    return res_value, res_colIndex, res_rowPtr, res_n, res_m


value_t, index_t, rowptr_t, n_t, m_t = csr_transpose((value, index, rowptr, n, m))

print(value_t, len(value_t))
print(index_t, len(index_t))
print(rowptr_t, len(rowptr_t))

num_nodes_t = len(rowptr_t) - 1
print(num_nodes_t)
csr_mtx_t = csr_matrix((value_t, index_t, rowptr_t), shape=(n_t, m_t))
dense_matrix_t = csr_mtx_t.toarray()
print(dense_matrix_t)

G = nx.DiGraph()
G.add_edges_from(edge_list)
pagerank = nx.pagerank(G, alpha=0.9)
print("PageRank 결과:")
for node, rank in list(pagerank.items())[:5]:
    print(f"{node}: {rank}")

gg = nx.DiGraph()
gg.add_edges_from(process_file("web-Google.txt"))
pg = nx.pagerank(gg, alpha=0.9)

top_5 = sorted(pg.items(), key=lambda x: x[1], reverse=True)[:10]

print("PageRank 결과:")
for node, rank in list(pg.items())[:10]:
    print(f"{node}: {rank}")

print("PageRank 상위 5개 결과:")
for node, rank in top_5:
    print(f"Node {node}: {rank}")

print("\nGoogle 데이터셋의 PageRank 결과 (노드 ID 순서):")
for node in list(sorted(pg.keys()))[:10]:
    print(f"Node {node}: {pg[node]}")

def read_pagerank_file(file_path):
    # 파일에서 직접 구현한 PageRank 결과 읽기
    pagerank_dict = {}
    with open(file_path, "r") as f:
        for line in f:
            node, value = line.strip().split("\t")
            pagerank_dict[int(node)] = float(value)
    return pagerank_dict

manual_pagerank = read_pagerank_file("pagerank.txt")

manual_values = np.array([manual_pagerank[node] for node in manual_pagerank])
networkx_values = np.array([pg[node] for node in sorted(pg.keys())])

difference = np.linalg.norm(manual_values - networkx_values)
print(f"벡터 차이 (L2 norm): {difference:.6e}")