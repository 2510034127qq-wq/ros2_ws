"""Prediction-based revisit scheduling with cooldown and exploration fairness."""
from dataclasses import dataclass
import math


@dataclass
class RevisitParams:
    stale_s: float = 8.0
    cooldown_s: float = 15.0
    prediction_horizon_s: float = 5.0
    max_consecutive: int = 2
    max_age_s: float = 90.0


class RevisitScheduler:
    def __init__(self, params=None):
        self.params=params or RevisitParams()
        self.last_selected={}
        self.consecutive=0

    def select(self, sources, now_s, robot_xy, reachable=None):
        p=self.params
        if self.consecutive>=p.max_consecutive:
            self.consecutive=0
            return None
        candidates=[]
        for s in sources:
            age=float(s.get('age_s',0))
            key=s['id']
            if s.get('status')=='suppressed' or age<p.stale_s or age>p.max_age_s:
                continue
            if now_s-self.last_selected.get(key,-math.inf)<p.cooldown_s:
                continue
            # Published positions are predictions at their own timestamp. Only
            # predict forward by reception latency, not by last-observation age.
            horizon=min(float(s.get('message_age_s',0)),p.prediction_horizon_s)
            x=s['x']+horizon*s.get('vx',0); y=s['y']+horizon*s.get('vy',0)
            if reachable is not None and not reachable(*robot_xy,x,y):
                continue
            value=s.get('probability',0)*max(s.get('strength',0),1)
            uncertainty=max(s.get('covariance_xx',1)+s.get('covariance_yy',1),.01)
            priority=value*uncertainty/(1+math.hypot(x-robot_xy[0],y-robot_xy[1]))
            candidates.append((priority,key,x,y))
        if not candidates:
            self.consecutive=0
            return None
        priority,key,x,y=max(candidates)
        self.last_selected[key]=now_s
        self.consecutive+=1
        return dict(x=x,y=y,score=priority,reason='predicted_revisit',source_id=key)
