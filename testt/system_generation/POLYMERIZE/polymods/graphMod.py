import sys
import numpy as np

class Graph():

    def __init__(self,adjmat):
        self.V     = adjmat.shape[0]
        self.graph = adjmat
        #self.graph = [[0 for column in range(vertices)]
        #            for row in range(vertices)]

    def printSolution(self, dist):
        print ("Vertex tDistance from Source")
        for node in range(self.V):
            print (node, "t", dist[node])

    # A utility function to find the vertex with
    # minimum distance value, from the set of vertices
    # not yet included in shortest path tree
    def minDistance(self, dist, sptSet):

        # Initilaize minimum distance for next node
        min = sys.maxsize

        # Search not nearest vertex not in the
        # shortest path tree
        for v in range(self.V):
            if dist[v] < min and sptSet[v] == False:
                min = dist[v]
                min_index = v

        return min_index

    # Funtion that implements Dijkstra's single source
    # shortest path algorithm for a graph represented
    # using adjacency matrix representation
    def dijkstra(self, src,dest=None):

        dist = [sys.maxsize] * self.V
        dist[src] = 0
        sptSet = [False] * self.V

        for cout in range(self.V):

            # Pick the minimum distance vertex from
            # the set of vertices not yet processed.
            # u is always equal to src in first iteration
            u = self.minDistance(dist, sptSet)

            # Put the minimum distance vertex in the
            # shotest path tree
            sptSet[u] = True

            # Update dist value of the adjacent vertices
            # of the picked vertex only if the current
            # distance is greater than new distance and
            # the vertex in not in the shotest path tree
            for v in range(self.V):
                if self.graph[u][v] > 0 and \
                     sptSet[v] == False and \
                     dist[v] > dist[u] + self.graph[u][v]:
                    dist[v] = dist[u] + self.graph[u][v]

        if dest is None:
          return np.array(dist)
        elif isinstance(dest,int):
          return dist[dest]
        elif isinstance(dest,list):
          return np.array([dist[i] for i in dest])
        else:
          print("Unrecognized data type as destiination in call to graphs.dijkstra")
          print("Recognized types are...")
          print("--None (returns all distances)")
          print("--int (returns distance from source to dest node)")
          print("--list (returns distances from source to all nodes in list)")
          exit()
