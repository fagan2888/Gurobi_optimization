from heapq import heappop, heappush


def dijkstra(graph, weight, source=0, target=None):
    n = len(graph)
    assert all(weight[u][int(v)] >= 0 for u in range(n) for v in graph[u])
    prec = [None] * n
    black = [False] * n
    dist = [float('inf')] * n
    dist[source] = 0
    heap = [(0, source)]
    while heap:
        dist_node, node = heappop(heap)
        if not black[node]:
            black[node] = True
            if node == target:
                break
            for neighbor in graph[node]:
                neighbor = int(neighbor)
                dist_neighbor = dist_node + weight[node][neighbor]
                if dist_neighbor < dist[neighbor]:
                    dist[neighbor] = dist_neighbor
                    prec[neighbor] = node
                    heappush(heap, (dist_neighbor, neighbor))
    return dist, prec

if __name__ == '__main__':
    graph = [[1], [0, 2, 3], [3], [3]]
    weight = [[0, 3, 0, 0], [0, 3, 4, 2], [0, 0, 0, 0], [0, 0, 0, 0]]
    dist, prec = dijkstra(graph, weight)