from __future__ import annotations
import networkx as nx
from .models import PairResult

def build_similarity_graph(results:list[PairResult],threshold:float=0.70)->nx.Graph:
    graph=nx.Graph()
    for result in results:
        graph.add_node(result.student_a); graph.add_node(result.student_b)
        score=float(result.model.get("score",0.0))
        if score>=threshold:
            graph.add_edge(result.student_a,result.student_b,score=score,language=result.evidence.get("language"))
    return graph

def relationship_groups(results:list[PairResult],threshold:float=0.70)->list[list[str]]:
    graph=build_similarity_graph(results,threshold)
    return [sorted(group) for group in nx.connected_components(graph) if len(group)>1]
