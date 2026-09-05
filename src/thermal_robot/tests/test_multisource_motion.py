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


def test_invalid_slow_observation_preserves_last_good_posterior():
    b=SourceBelief(BeliefParams(budget_ms=1000))
    assert b.update([det(1.)],1.)
    before=b.clusters[0].motion.state.copy()
    assert not b.update([det(float('nan'))],2.)
    assert b.health=='invalid:ValueError' and b.stamp_s==1.
    assert np.array_equal(before,b.clusters[0].motion.state)
    assert b.update([det(1.2)],3.) and b.health=='ready'


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


def test_field_partial_footprint_is_not_a_source_peak():
    tracker=SourceTrackerCore(motion_model='kalman',max_detection_age_s=1.5)
    y,x=np.indices((40,40));temp=22+30*np.exp(-((x-22)**2+(y-20)**2)/32)
    confidence=np.zeros_like(temp);confidence[:, :20]=1
    assert tracker.extract_detections(temp,confidence,.25,0,0,np.zeros_like(temp))==[]
    confidence[:,:]=1
    detections=tracker.extract_detections(temp,confidence,.25,0,0,np.zeros_like(temp))
    assert len(detections)==1 and detections[0].x==pytest.approx(5.625,abs=.1)


def test_noisy_flat_surface_makes_one_detection():
    tracker=SourceTrackerCore(motion_model='kalman',max_detection_age_s=1.5)
    tracker.measurement_type='surface_radiance'
    temp=np.full((40,40),22.);temp[18:22,10:20]=50+np.random.default_rng(1).normal(0,.1,(4,10))
    detections=tracker.extract_detections(temp,np.ones_like(temp),.25,0,0,np.zeros_like(temp))
    assert len(detections)==1


def test_gp_posterior_reduces_uncertainty_and_checks_reachability():
    from thermal_motion_controller.gp_ucb import GaussianProcessUCB
    gp=GaussianProcessUCB();gp.fit([[0,0],[1,0]],[20,18])
    mean,var=gp.predict([[0,0],[20,20]])
    assert mean[0]>15 and var[0]<var[1]*.1
    m=dict(width=10,height=10,origin_x=0.,origin_y=0.,resolution=1.,
           confidence=np.ones((10,10)),temperature_mean=np.ones((10,10))*30,
           last_seen_age_s=np.zeros((10,10)))
    assert gp.select(m,(0,0),2,10,lambda *args:False) is None
    assert gp.select(m,(0,0),2,10,lambda *args:True) is not None


def test_dynamic_clearance_evidence_ages_and_candidate_probability_counts():
    from thermal_motion_controller.clearance import ClearanceParams,clearance_with_history
    p=ClearanceParams();state=np.ones((4,4),dtype=np.uint8)*2;sectors=np.full((4,4),15,dtype=np.uint8)
    fresh,_=clearance_with_history(state,sectors,np.zeros((4,4)),1.,p)
    stale,_=clearance_with_history(state,sectors,np.ones((4,4))*60,1.,p)
    candidate,_=clearance_with_history(state,sectors,np.zeros((4,4)),1.,p,candidate_probabilities=[.8])
    assert stale<fresh and candidate==pytest.approx(.2*fresh)


def test_clearance_conditions_all_sector_misses_on_one_source_amplitude():
    from thermal_motion_controller.clearance import ClearanceParams,clearance_with_history
    p=ClearanceParams(source_rate_per_m2=1.,amplitude_prior_mean_c=4.,source_birth_rate_m2_s=0.)
    _,mass=clearance_with_history(np.array([[2]]),np.array([[15]],dtype=np.uint8),
        np.array([[0.]]),1.,p,detection_distance_m=np.array([[8.]]))
    rng=np.random.default_rng(13)
    amplitude=4.+rng.exponential(4.,200000)
    detection=.7/(1+np.exp(-(amplitude/(1+(8/5)**2)-4.)))
    missed=(rng.random((len(amplitude),4))>=detection[:,None]).all(axis=1)
    assert mass==pytest.approx(missed.mean(),abs=.006)


def test_belief_field_fit_infers_amplitude_and_scale():
    y,x=np.indices((41,41));wx=(x+.5)*.25;wy=(y+.5)*.25
    temp=22+30*np.exp(-((wx-5)**2+(wy-5)**2)/(2*1.2**2))
    b=SourceBelief(BeliefParams(budget_ms=1000))
    m=dict(temperature_mean=temp,confidence=np.ones_like(temp),resolution=.25,
           origin_x=0.,origin_y=0.,ambient=22.,last_seen_age_s=np.zeros_like(temp))
    for t in range(8):b.update([det(5,5,30)],float(t),field_snapshot=m)
    c=b.clusters[0]
    assert c.amplitude==pytest.approx(30,abs=5) and c.sigma==pytest.approx(1.2,abs=.3)


@pytest.mark.parametrize('strategy,mode,health,stamp,now,receipt,expected',[
    ('dual','online','ready',10.,11.,.1,True),
    ('fast','online','ready',10.,11.,.1,False),
    ('gp_ucb','online','ready',10.,11.,.1,False),
    ('dual','shadow','ready',10.,11.,.1,False),
    ('dual','off','ready',10.,11.,.1,False),
    ('dual','online','over_budget',10.,11.,.1,False),
    ('dual','online','error:ValueError',10.,11.,.1,False),
    ('dual','online','ready',10.,14.,.1,False),
    ('dual','online','ready',10.,11.,4.,False),
    ('dual','online','ready',12.,11.,.1,False),
    ('dual','online','ready',float('nan'),11.,.1,False),
])
def test_slow_policy_blocks_stale_faulty_and_baseline_feedback(strategy,mode,health,stamp,now,receipt,expected):
    from thermal_motion_controller.runtime_policy import slow_output_usable
    assert slow_output_usable(strategy,mode,health,stamp,now,receipt)==expected


def test_surface_approach_selects_free_footprint_and_rejects_unknown_space():
    from thermal_motion_controller.runtime_policy import surface_approach_waypoint
    from thermal_field_reconstructor.visibility import OccupancyView,known_free_at
    grid=np.zeros((60,60),dtype=np.int16)
    grid[:,20:23]=100
    view=OccupancyView(-3.,-3.,.1,grid)
    target=surface_approach_waypoint((-2.,0.),(0.,0.),1.2,view)
    assert target is not None and known_free_at(view,np.array([target[0]]),np.array([target[1]]))[0]
    assert np.linalg.norm(target)==pytest.approx(1.2)
    grid[:]=-1
    assert surface_approach_waypoint((-2.,0.),(0.,0.),1.2,view) is None


def test_merge_requires_persistent_overlap_and_preserves_confirmed_identities():
    from thermal_motion_controller.belief import SourceCluster
    from thermal_motion_controller.motion_filter import MotionFilter
    b=SourceBelief(BeliefParams(merge_hits=3))
    b.clusters=[SourceCluster(str(i),MotionFilter(np.array([x,0.,0.,0.])),.4,20.,.6,0.)
                for i,x in enumerate((0.,.2))]
    b._merge();b._merge()
    assert len(b.clusters)==2
    b._merge()
    assert len(b.clusters)==1 and b.events[-1]['kind']=='merge'
    assert np.linalg.eigvalsh(b.clusters[0].motion.covariance).min()>0
    other=SourceCluster('other',MotionFilter(np.array([.1,0.,0.,0.])),.99,20.,.6,0.,confirmed=True)
    b.clusters[0].confirmed=True;b.clusters.append(other)
    for _ in range(5):b._merge()
    assert len(b.clusters)==2


def test_five_moving_sources_with_late_birth_death_and_rebirth():
    b=SourceBelief(BeliefParams(budget_ms=1000,survival_time_s=1000))
    events=[]
    for t in range(30):
        detections=[det(5*i+.1*t,2.*(i%2),20.+i) for i in range(5)
                    if not (i==4 and t<5) and not (i==0 and 12<=t<21)]
        assert b.update(detections,float(t),lambda x,y:1.)
        events.extend(e['kind'] for e in b.events)
        assert len(b.clusters)<=5
        assert np.isfinite(b.cardinality()).all() and b.cardinality().sum()==pytest.approx(1.)
    assert np.argmax(b.cardinality())==5 and b.cardinality()[5]>.95
    assert events.count('birth')==6 and 'death' in events
    assert all(c.motion.state[2]==pytest.approx(.1,abs=.04) for c in b.clusters)
