"""Motion, association, cardinality and revisit requirements on synthetic evidence."""
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
for pkg in ('thermal_motion_controller','thermal_field_reconstructor'):
    sys.path.insert(0,str(ROOT/pkg))
from thermal_motion_controller.source_tracking import SourceTrackerCore, SourceDetection
from thermal_motion_controller.belief import SourceBelief, BeliefParams
from thermal_motion_controller.revisit import RevisitScheduler, RevisitParams


def det(x,y=0.,a=20.):
    return SourceDetection(x,y,a,.95,.6)


def test_constant_velocity_reacquisition_keeps_identity():
    tracker=SourceTrackerCore(motion_model='kalman',stale_after_s=2.,confirm_observations=3)
    for t in range(12):
        tracks=tracker.update([det(.3*t)],float(t))
    tr=tracks[0]; key=tr.track_id
    assert tr.status=='confirmed'
    assert tr.vx==pytest.approx(.3,abs=.04)
    for t in (12.,13.,14.):
        tracker.update([],t)
    assert tr.status=='stale'
    assert tr.x==pytest.approx(4.2,abs=.15)
    tracker.update([det(4.5)],15.)
    assert len(tracker.tracks)==1 and tr.track_id==key
    assert tr.status=='confirmed' and tr.reacquisitions==1
    assert np.linalg.eigvalsh(tracker._filters[key].covariance).min()>0


def test_crossing_tracks_do_not_merge():
    tracker=SourceTrackerCore(motion_model='kalman',merge_radius_m=.2,confirm_observations=3)
    for t in range(20):
        tracks=tracker.update([det(-2+.25*t,.1,15),det(2-.25*t,-.1,35)],float(t))
    live=[t for t in tracks if t.status=='confirmed']
    assert len(live)==2
    assert sorted(t.vx for t in live)==pytest.approx([-.25,.25],abs=.05)
    assert next(t for t in live if t.strength<25).vx>0


def test_belief_count_birth_occlusion_death_and_information():
    b=SourceBelief(BeliefParams(budget_ms=1000,survival_time_s=1000))
    for t in range(6):
        assert b.update([det(0),det(4)],float(t),lambda x,y:1.)
    pmf=b.cardinality()
    assert np.argmax(pmf)==2 and pmf[2]>.95 and sum(pmf)==pytest.approx(1)
    assert b.information_gain([[0,0],[100,100]])[0]>b.information_gain([[0,0],[100,100]])[1]
    p=b.clusters[0].probability
    b.update([],6.,lambda x,y:0.)
    assert b.clusters[0].probability>.99*p
    for t in range(7,20):
        b.update([],float(t),lambda x,y:1.)
    assert len(b.clusters)==0
    assert b.clearance(0)==pytest.approx(1.)


def test_belief_residual_birth_and_transactional_budget():
    b=SourceBelief(BeliefParams(budget_ms=1000))
    b.update([det(0)],0.,birth_detections=[])
    assert not b.clusters
    b.update([det(0)],1.,birth_detections=[det(0)])
    assert len(b.clusters)==1
    stamp=b.stamp_s
    b.params.budget_ms=-1
    assert not b.update([det(1)],2.)
    assert b.stamp_s==stamp and b.health=='over_budget'


def test_belief_split_birth_does_not_erase_parent():
    b=SourceBelief(BeliefParams(budget_ms=1000))
    for t in range(6): b.update([det(0)],float(t))
    for t in range(6,12): b.update([det(0),det(1.)],float(t))
    assert len(b.clusters)==2 and np.argmax(b.cardinality())==2
    assert b.clusters[1].parent==b.clusters[0].label


def test_revisit_prediction_cooldown_and_fairness():
    r=RevisitScheduler(RevisitParams(max_consecutive=1,cooldown_s=10))
    sources=[dict(id='a',x=5.,y=0.,vx=1.,vy=0.,age_s=10.,message_age_s=1.,
                  probability=.9,strength=20.,status='stale')]
    assert r.select(sources,20.,(0,0),lambda *args:False) is None
    target=r.select(sources,20.,(0,0))
    assert target['x']==6.
    assert r.select(sources,21.,(0,0)) is None
    assert r.select(sources,22.,(0,0)) is None
