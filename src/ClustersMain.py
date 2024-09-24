import networkx as nx
import community  # Louvain community detection
import Service.SDP import _SDP
from Service.TwoCoreReduction import TwoCoreReduction


class SDP1:
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.results = []
        self.clusters = []

    def runClustersDetection(self):
        two2CoreReduction = TwoCoreReduction(iNetwork.graph).run2CoreReduction()
        after_nodes=len(two2CoreReduction)

        try:
            sdpObj=SDP(two2CoreReduction)
            sdpObj.runSDP()
            bestCut=sdpObj.getBestCut()
            showGraph(two2CoreReduction,bestCut)


            clusters=[]

            for user in bestCut.cut:
                values=user.stance
                Users.append(user.userId)
                if value < float(0):
                    partition.append(0)
                else:
                    partition.append(1)
G = nx.Graph()  # Construct your graph
sdp = SDP(G)
sdp.runClustersDetection()
print(sdp)
