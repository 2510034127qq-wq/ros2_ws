"""Bounded exact GP regression with an RBF kernel and UCB action selection."""
from dataclasses import dataclass
import numpy as np
from .planning import PlannerTarget


@dataclass
class GPParams:
    max_samples: int = 64
    max_candidates: int = 256
    length_scale_m: float = 2.0
    signal_std_c: float = 12.0
    noise_std_c: float = 1.0
    beta: float = 2.0
    travel_weight: float = .2


class GaussianProcessUCB:
    def __init__(self,params=None):
        self.params=params or GPParams()
        self.x=np.empty((0,2));self.alpha=None;self.chol=None

    def kernel(self,x,y):
        return self.params.signal_std_c**2*np.exp(-np.sum(
            (np.asarray(x)[:,None,:]-np.asarray(y)[None,:,:])**2,axis=2)
            /(2*self.params.length_scale_m**2))

    def fit(self,points,values):
        x=np.asarray(points,dtype=float).reshape(-1,2);y=np.asarray(values,dtype=float)
        valid=np.isfinite(x).all(axis=1)&np.isfinite(y)
        x,y=x[valid],y[valid]
        if len(x)>self.params.max_samples:
            # Deterministic spatially broad subset plus strong observations.
            n=self.params.max_samples
            selected=list(np.linspace(0,len(x)-1,n//2,dtype=int))
            selected+=list(np.argsort(y)[::-1])
            selected=list(dict.fromkeys(selected))[:n]
            x,y=x[selected],y[selected]
        self.x=x
        if not len(x): self.alpha=self.chol=None;return
        k=self.kernel(x,x)+np.eye(len(x))*(self.params.noise_std_c**2+1e-6)
        self.chol=np.linalg.cholesky(k)
        self.alpha=np.linalg.solve(self.chol.T,np.linalg.solve(self.chol,y))

    def predict(self,points):
        points=np.asarray(points,dtype=float).reshape(-1,2)
        if self.alpha is None:
            return np.zeros(len(points)),np.full(len(points),self.params.signal_std_c**2)
        k=self.kernel(self.x,points)
        v=np.linalg.solve(self.chol,k)
        return k.T@self.alpha,np.maximum(1e-9,self.params.signal_std_c**2-np.sum(v*v,axis=0))

    def select(self,snapshot,robot_xy,min_d,max_d,reachable=None,candidate_mask=None,ambient=22.):
        m=snapshot;yy,xx=np.indices((m['height'],m['width']))
        xy=np.column_stack((m['origin_x']+(xx.ravel()+.5)*m['resolution'],
                            m['origin_y']+(yy.ravel()+.5)*m['resolution']))
        valid=np.asarray(m['confidence']).ravel()>.1
        if 'last_seen_age_s' in m: valid &= np.asarray(m['last_seen_age_s']).ravel()<5.
        self.fit(xy[valid],np.asarray(m['temperature_mean']).ravel()[valid]-ambient)
        distance=np.linalg.norm(xy-np.asarray(robot_xy),axis=1)
        candidates=(distance>=min_d)&(distance<=max_d)
        if candidate_mask is not None: candidates &= np.asarray(candidate_mask).ravel()
        indices=np.flatnonzero(candidates)
        if not len(indices): return None
        indices=indices[np.linspace(0,len(indices)-1,min(len(indices),self.params.max_candidates),dtype=int)]
        mean,var=self.predict(xy[indices])
        scores=mean+self.params.beta*np.sqrt(var)-self.params.travel_weight*distance[indices]
        for k in np.argsort(scores)[::-1]:
            x,y=xy[indices[k]]
            if reachable is None or reachable(*robot_xy,x,y):
                return PlannerTarget(float(x),float(y),float(scores[k]),'gp_ucb')
        return None
