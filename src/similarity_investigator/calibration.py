from __future__ import annotations
from collections import defaultdict
from statistics import median
from .models import PairResult

def _percentile(value:float,values:list[float])->float:
    if len(values)<=1:return 0.5
    less=sum(item<value for item in values); equal=sum(item==value for item in values)
    return (less+0.5*equal)/(len(values)-1)

def calibrate_results(results:list[PairResult])->None:
    groups=defaultdict(list)
    for result in results: groups[result.evidence.get("language","Unknown")].append(result)
    for language,items in groups.items():
        scores=[float(item.model.get("score",0.0)) for item in items]
        med=median(scores)
        deviations=[abs(score-med) for score in scores]
        mad=median(deviations)
        scale=1.4826*mad
        for item in items:
            score=float(item.model.get("score",0.0))
            item.features["corpus_percentile"]=_percentile(score,scores)
            item.features["corpus_mad_z"]=0.0 if scale==0 else (score-med)/scale
            item.evidence["calibration"]={"language":language,"median":med,"mad":mad,"absolute_score":score}
