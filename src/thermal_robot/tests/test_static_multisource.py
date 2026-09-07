"""Stationary association, cardinality and navigation requirements on synthetic evidence."""
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
for pkg in ('thermal_motion_controller','thermal_field_reconstructor'):
    sys.path.insert(0,str(ROOT/pkg))
from thermal_motion_controller.source_tracking import SourceTrackerCore, SourceDetection
from thermal_motion_controller.belief import SourceBelief, BeliefParams


def det(x,y=0.,a=20.):
    return SourceDetection(x,y,a,.95,.6)


def test_position_filter_smooths_noise_without_drifting_during_occlusion():
    from thermal_motion_controller.position_filter import PositionFilter
    filt = PositionFilter(np.array([4, 2]))
    rng = np.random.default_rng(101)
    raw, estimated = [], []
    for t in range(1, 51):
        xy = np.array([4., 2.]) + rng.normal(0, .2, 2)
        filt.predict(float(t))
        filt.correct(xy, .04)
        raw.append(np.linalg.norm(xy-[4, 2]))
        estimated.append(np.linalg.norm(filt.state-[4, 2]))
    assert np.mean(estimated) < np.mean(raw)*.6
    last, cov = filt.state.copy(), filt.covariance.copy()
    filt.predict(110.)
    assert np.array_equal(filt.state, last)
    assert np.trace(filt.covariance) > np.trace(cov)
    assert np.linalg.eigvalsh(filt.covariance).min() > 0
    with pytest.raises(ValueError):
        filt.predict(109.)


@pytest.mark.parametrize('estimator', ['legacy','kalman'])
def test_static_tracker_keeps_identity_across_long_absence(estimator):
    tracker = SourceTrackerCore(estimator_model=estimator, confirm_observations=3)
    for stamp in range(12): tracker.update([det(3.+.05*np.sin(stamp))], float(stamp))
    track = tracker.tracks[0]
    key, position, probability = track.track_id, (track.x,track.y), track.existence_probability
    tracker.update([], 1200.)
    assert tracker.tracks == [track] and track.status == 'confirmed'
    assert (track.x,track.y) == position and track.existence_probability == probability
    assert track.last_seen_s == 11.
    tracker.update([det(3.)],1201.)
    assert tracker.tracks == [track] and track.track_id == key and track.status == 'confirmed'


def test_five_static_sources_with_incremental_visibility_have_stable_labels():
    b = SourceBelief(BeliefParams(budget_ms=1000))
    tracker = SourceTrackerCore(estimator_model="kalman")
    keys = None
    for t in range(30):
        # The fifth heater exists from the start but is initially outside view.
        detections = [det(4*i+.03*np.sin(t+i), 2.*(i%2), 20.+i)
                      for i in range(5) if i < 4 or t >= 5]
        assert b.update(tracker.update(detections,float(t)),float(t))
        assert len(b.clusters) <= 5
        assert b.cardinality().sum() == pytest.approx(1.)
        if t == 15:
            keys = [c.label for c in b.clusters]
    assert keys == [c.label for c in b.clusters]
    assert b.cardinality()[5] > .95
    assert all(c.position.state.shape == (2,) for c in b.clusters)


@pytest.mark.parametrize('unsupported', [
    {'motion': 'linear'}, {'active_schedule': [[0, 10]]},
    {'motion_params': {'vx': 1.}}, {'strength_drift': {'rate_per_s': 1.}},
])
def test_removed_dynamic_scenarios_fail_explicitly(tmp_path, unsupported):
    import yaml
    sys.path.insert(0, str(ROOT/'thermal_sensor_sim'))
    from thermal_sensor_sim.scenario import load_scenario_file
    path = tmp_path/'unsupported.yaml'
    path.write_text(yaml.safe_dump({'sources': [dict(id='a', xy=[0,0], **unsupported)]}))
    with pytest.raises(ValueError, match='stationary'):
        load_scenario_file(str(path))


@pytest.mark.parametrize('immediate,count', [(False,0),(False,1),(False,3),(True,1),(True,3)])
def test_legacy_policy_leaves_peak_without_declaring_search_complete(immediate, count):
    # Exercise the real timer's transition without a ROS executor or Gazebo.
    import ast
    import math
    from types import SimpleNamespace
    path = ROOT/'thermal_motion_controller/thermal_motion_controller/controller_node.py'
    tree = ast.parse(path.read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_timer_cb')
    scope = {'math':math, 'time':SimpleNamespace(monotonic=lambda:100.), 'Twist':lambda:None}
    scope.update({n.targets[0].id:n.value.value for n in tree.body
                  if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)
                  and n.targets[0].id.startswith('STATE_') and isinstance(n.value,ast.Constant)})
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),scope)
    node = SimpleNamespace(_t0=0., _current_temp=lambda:30., _grad_mag=lambda:1.,
        _temp_rise=lambda:8., _found_sources=[(0.,0.,30.)]*count,
        _update_world_pos_from_tf=lambda:False, _sensor_model='a', _strategy_mode='full',
        _update_armed=lambda t:None, _state='AT_PEAK', _state_t=0., _peak_hold=3.,
        _pub=SimpleNamespace(publish=lambda msg:None), _post_confirm_immediate_departure=immediate,
        _wx=1., _wy=2., _escape_yaw=lambda:0., _ambient_est=22., _temp_win=[],
        get_logger=lambda:SimpleNamespace(info=lambda msg:None))
    node._start_post_confirm_departure=lambda now,n:setattr(node,'_state','DEPARTURE')
    scope['_timer_cb'](node)
    assert node._state == ('DEPARTURE' if immediate and count else 'RELOCATE')
    if node._state=='RELOCATE':
        assert node._move_start == (1.,2.) and node._state_t == 100.






def test_belief_follows_registry_confirmation_and_candidate_removal():
    tracker=SourceTrackerCore(estimator_model='kalman',candidate_timeout_s=3.)
    b=SourceBelief(BeliefParams(budget_ms=1000))
    for t in range(8):
        b.update(tracker.update([det(0),det(4)],float(t)),float(t))
    keys=[c.label for c in b.clusters]
    assert np.argmax(b.cardinality())==2 and b.cardinality()[2]>.95
    b.update(tracker.update([det(10)],8.),8.)
    assert len(b.clusters)==3
    b.update(tracker.update([],1200.),1200.)
    assert [c.label for c in b.clusters]==keys
    assert all(c.confirmed for c in b.clusters)
    assert [c.last_seen_s for c in b.clusters]==[7.,7.]


def test_belief_registry_update_is_transactional_on_budget_failure():
    tracker=SourceTrackerCore(estimator_model='kalman')
    b=SourceBelief(BeliefParams(budget_ms=1000))
    assert b.update(tracker.update([det(0)],0.),0.)
    key=b.clusters[0].label
    b.params.budget_ms=-1
    assert not b.update(tracker.update([det(0),det(4)],1.),1.)
    assert b.stamp_s==0. and b.health=='over_budget' and [c.label for c in b.clusters]==[key]


def test_invalid_slow_observation_preserves_last_good_posterior():
    import copy
    tracker=SourceTrackerCore(estimator_model='kalman')
    b=SourceBelief(BeliefParams(budget_ms=1000))
    registry=tracker.update([det(1.)],1.)
    assert b.update(registry,1.)
    before=b.clusters[0].position.state.copy()
    broken=copy.deepcopy(registry);broken[0].x=float('nan')
    assert not b.update(broken,2.)
    assert b.health=='invalid:ValueError' and b.stamp_s==1.
    assert np.array_equal(before,b.clusters[0].position.state)
    assert b.update(tracker.update([det(1.2)],3.),3.) and b.health=='ready'


def test_belief_uses_canonical_ids_without_spatial_rematching():
    import copy
    tracker=SourceTrackerCore(estimator_model='kalman')
    b=SourceBelief(BeliefParams(budget_ms=1000))
    for t in range(8):b.update(tracker.update([det(0),det(4)],float(t)),float(t))
    registry=copy.deepcopy(tracker.tracks)
    # Registry is authoritative even when a map correction puts entries close together.
    registry[1].x=.1
    assert b.update(registry,8.)
    assert [c.label for c in b.clusters]==[tr.track_id for tr in registry]
    assert all(c.confirmed for c in b.clusters)




def test_field_partial_footprint_is_not_a_source_peak():
    tracker=SourceTrackerCore(estimator_model='kalman',max_detection_age_s=1.5)
    y,x=np.indices((40,40));temp=22+30*np.exp(-((x-22)**2+(y-20)**2)/32)
    confidence=np.zeros_like(temp);confidence[:, :20]=1
    assert tracker.extract_detections(temp,confidence,.25,0,0,np.zeros_like(temp))==[]
    confidence[:,:]=1
    detections=tracker.extract_detections(temp,confidence,.25,0,0,np.zeros_like(temp))
    assert len(detections)==1 and detections[0].x==pytest.approx(5.625,abs=.1)


def test_noisy_flat_surface_makes_one_detection():
    tracker=SourceTrackerCore(estimator_model='kalman',max_detection_age_s=1.5)
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






def test_belief_field_fit_infers_amplitude_and_scale():
    y,x=np.indices((41,41));wx=(x+.5)*.25;wy=(y+.5)*.25
    temp=22+30*np.exp(-((wx-5)**2+(wy-5)**2)/(2*1.2**2))
    b=SourceBelief(BeliefParams(budget_ms=1000))
    m=dict(temperature_mean=temp,confidence=np.ones_like(temp),resolution=.25,
           origin_x=0.,origin_y=0.,ambient=22.,last_seen_age_s=np.zeros_like(temp))
    tracker=SourceTrackerCore(estimator_model="kalman")
    for t in range(8):b.update(tracker.update([det(5,5,30)],float(t)),float(t),field_snapshot=m)
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
    assert 1.2-1e-6<=np.linalg.norm(target)<2.  # A safe intermediate step is valid.
    grid[:]=0
    target=surface_approach_waypoint((-2.,0.),(0.,0.),1.2,view)
    assert np.linalg.norm(target)==pytest.approx(1.2)
    grid[:]=-1
    assert surface_approach_waypoint((-2.,0.),(0.,0.),1.2,view) is None


def test_surface_approach_can_advance_before_distant_standoff_is_mapped():
    from thermal_motion_controller.runtime_policy import surface_approach_waypoint
    from thermal_field_reconstructor.visibility import OccupancyView,known_free_at
    grid=np.full((80,140),-1,dtype=np.int16)
    grid[20:60,30:65]=0  # Only the local patch, not the source at (7,0), is known.
    view=OccupancyView(-4.,-4.,.1,grid)
    target=surface_approach_waypoint((0.,0.),(7.,0.),1.2,view,max_step_m=2.)
    assert target is not None
    assert .5<np.linalg.norm(target)<=2.+1e-6
    assert np.linalg.norm(np.asarray(target)-[7.,0.])<6.75
    angles=np.linspace(0.,2*np.pi,24,endpoint=False)
    footprint=np.asarray(target)+.35*np.column_stack((np.cos(angles),np.sin(angles)))
    assert known_free_at(view,footprint[:,0],footprint[:,1]).all()
    grid[:,40:]=-1  # Only space behind the robot remains known: no progress.
    assert surface_approach_waypoint((0.,0.),(7.,0.),1.2,view) is None


def test_registry_rejects_duplicate_hot_faces_but_keeps_nearby_distinct_sources():
    tracker=SourceTrackerCore(estimator_model='kalman',merge_radius_m=.8,gate_m=1.25)
    for t in range(8): tracker.update([det(0),det(.3),det(1.8)],float(t))
    assert len(tracker.tracks)==2 and all(t.status=='confirmed' for t in tracker.tracks)
    keys=[t.track_id for t in tracker.tracks]
    tracker.update([],1000.)
    tracker.update([det(.2),det(1.7)],1001.)
    assert [t.track_id for t in tracker.tracks]==keys
