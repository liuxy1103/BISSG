"""graph_partition

    `Greedy Additive Edge Contraction 法 (GAEC)
    <https://arxiv.org/abs/1505.06973>`_ による Edge の階層的な縮約を行う.
      - Graph: Partition 集合 (P) と Edge 集合 (E) の組 (P, E).
      - Partition: Edge の縮約によって結合された Node の集合.  #通过缩约而结合的节点的集合。
      - Edge: Partition を結ぶ.それぞれに切断コスト w が定義されている.  #分别定义了切断成本w。

"""

import heapq
import numpy as np


class Partition:
    """Partition

     Nodeの集合とつながっているEdgeに関する情報を保持.
     各Nodeの集合を定義している.
     グラフの統合の際に、まとめるべきNodeとつなぎ直すEdgeを見つけるために必要となる.

    """

    def __init__(self, nodes):
        """__init__

            自身が持つNodeの集合を定義.

            Args:
                nodes (list): 自身のNodeの集合.中身の型の指定はない.

        """
        self.nodes = nodes
        self.links = {}
        self.merged = False

    def __repr__(self):
        return 'Partition(nodes={})'.format(self.nodes)

    def connect(self, edge):
        """connect

            自身とつながっているPartition及びEdgeをself.linksに格納.

            Args:
                edge (Edge): 自身とつながっているEdge.

        """
        pair = edge.get_pair(self)
        self.links[id(pair)] = (edge, pair)

    def disconnect(self, edge):
        """disconnect

            つながっているedgeを切り離す. # 切断连接着的edge。

            Args:
                edge (Edge): 自身とつながっているedge.

        """
        pair = edge.get_pair(self)
        self.links.pop(id(pair))

    def merge_nodes(self, partition):
        """merge_nodes

            引数に指定したPartitionを自身のNodeに取り込む.

            Args:
                partition (Partition): 自身とつながっているPartition.

        """
        self.nodes += partition.nodes
        partition.nodes = []
        partition.merged = True

    def get_edge(self, partition):
        """get_edge

            自身と引数で指定したPartitionを繋げているEdgeを返す.
            #返回连接自身和自变量指定的Partition的Edge。

            Args:
                partition (Partition): 自身とつながっているPartition.

        """
        if id(partition) not in self.links:
            return None
        return self.links[id(partition)][0]


class Edge:
    """Edge

        Partition間のEdgeとその重みを保持.
        Partition間の関係を示す.

    """

    def __init__(self, partition0, partition1, weight):
        """__init__

            Edgeの定義.

            Args:
                partition0 (Partition): EdgeにつながっているPartition.
                partition１ (Partition): Edgeにつながっているもう一方のPartition.
                weight (float): Edgeの重み.

        """
        self.pair = (partition0, partition1)
        self.weight = weight
        self.removed = False
        partition0.connect(self)
        partition1.connect(self)

    def __lt__(self, edge):
        # priority que のため逆にしている
        return self.weight > edge.weight

    def __repr__(self):
        return 'Edge({}<=>{})'.format(*self.pair)

    def get_pair(self, partition):
        """get_pair

            Edgeでつながっている二つのPartitionのうち、引数ではないPartitionを返す.

            Args:
                partition (Partition): Edgeにつながっている引数ではないPartition.

        """
        if partition is self.pair[0]:
            return self.pair[1]
        else:
            return self.pair[0]

    def remove(self):
        """remove

            つながっているPartitionを切り、Edgeを消去する.

        """
        partition0, partition1 = self.pair
        partition0.disconnect(self)
        partition1.disconnect(self)
        self.removed = True

    def contract(self):
        """contract

            Partitionを新しくつなげてnew_edgeを作成し、重みを更新する.
            重新连接Partition，创建new_edge来更新权重。

            Returns:
                Edge: 新しく作成したEdge.

        """
        # import pdb
        # pdb.set_trace()
        partition0, partition1 = self.pair
        new_edge = []

        # edgeが多くつながっているNodeを元Nodeにする.  将edge连接较多的节点作为原节点。
        if len(partition1.links) > len(partition0.links):
            partition0, partition1 = partition1, partition0

        for edge12, partition2 in list(partition1.links.values()):
            # すでにつながっている場合は何もしない.
            if partition0 is partition2:
                continue
            edge02 = partition0.get_edge(partition2)
            # つながっていない場合は新しくEdgeを作成.
            if edge02 is None:
                edge02 = Edge(partition0, partition2, 0)
                new_edge.append(edge02)

            # それぞれのEdgeの重みを更新.
            edge02.weight += edge12.weight
            edge12.remove()

        partition0.merge_nodes(partition1)
        self.remove()

        return new_edge


def greedy_additive(edges, partitions,thre=0):
    """greedy_additive

         与えられたEdge(edges)とNodeの集合(partitions)からグラフ分割を行う.
         从给定的Edge(edges)和Node的集合(partitions)进行图表分割。

        Args:
           edges (list): Edge.
           partitions (list): Nodeの集合(partition).

        Returns:
           list: グラフ分割後のEdge. 图表分割后的Edge。
           list: グラフ分割後のNodeの集合(partition). 图分割后节点的集合(partition)。

        Examples:
            >>> p0 = Partition([0])
            >>> p1 = Partition([1])
            >>> p2 = Partition([2])
            >>> p3 = Partition([3])
            >>> e01 = Edge(p0, p1, 1)
            >>> e12 = Edge(p1, p2, 2)
            >>> e23 = Edge(p2, p3, 3)
            >>> e30 = Edge(p3, p0, -10)

            >>> e, p = greedy_additive([e01, e12, e23, e30], [p0, p1, p2, p3])
            >>> e, p
                ([Edge(Partition(nodes=[0])<=>Partition(nodes=[1, 2, 3]))],
                 [Partition(nodes=[0]), Partition(nodes=[1, 2, 3])])
    """
    # import pdb
    # pdb.set_trace()

    heapq.heapify(edges)

    while edges:
        edge = heapq.heappop(edges)

        if edge.removed:
            continue

        # 全てのedgeの重みが0以下になったら終了. 如果所有edge的权重都低于0就结束。
        if edge.weight < thre:

            heapq.heappush(edges, edge)
            break

        new_edges = edge.contract()

        for new_edge in new_edges:
            heapq.heappush(edges, new_edge)

    # 結合して不要になったedgeとpartitionを取り除く. 结合去掉不必要的edge和partition。

    edges = list(filter(lambda e: not e.removed, edges))
    partitions = list(filter(lambda p: not p.merged, partitions))

    return edges, partitions



if __name__ == '__main__':
    E = []
    N = []
    for i in range(4):
        N.append(Partition([i]))
    # p1 = Partition([1])
    # p2 = Partition([2])
    # p3 = Partition([3])
    # p4 = Partition([4])
    e01 = Edge(N[1],N[0] , 0.1)
    e12 = Edge(N[1], N[2], 0.2)
    e23 = Edge(N[2], N[3], -1)
    # # e30 = Edge(p3, p0, -10)
    #
    m = greedy_additive([e01, e23, e12], N)
    print(m[0],m[1])