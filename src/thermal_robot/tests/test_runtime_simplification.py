"""Exercise changed runtime methods without starting a ROS executor."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace as NS
import time
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for package in ('thermal_motion_controller', 'thermal_field_reconstructor'):
    sys.path.insert(0, str(ROOT / package))
from thermal_motion_controller.belief import source_information_gain
from thermal_field_reconstructor import visibility


def method(package, module, name, **scope):
    path = ROOT / package / package / (module + '.py')
    node = next(n for n in ast.walk(ast.parse(path.read_text()))
                if isinstance(n, ast.FunctionDef) and n.name == name)
    for arg in node.args.args:
        arg.annotation = None
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    return scope[name]


@pytest.mark.parametrize('spawn,world', [((-6., 0.), (-4., 2.)), ((3., -2.), (4., -1.))])
def test_simulated_image_pose_matches_absolute_gazebo_odometry(spawn, world):
    import xml.etree.ElementTree as ET
    robot = ET.parse(ROOT / 'g1_description/urdf/g1_nav.urdf')
    assert robot.find('.//plugin[@name="diff_drive_controller"]/odometry_source').text == '1'
    callback = method('thermal_sensor_sim', 'sensor_node', '_odom_cb', math=math)
    node = NS(_spawn_x=spawn[0], _spawn_y=spawn[1])
    callback(node, NS(pose=NS(pose=NS(position=NS(x=world[0], y=world[1]),
        orientation=NS(w=math.cos(.3), z=math.sin(.3), x=0., y=0.)))))
    assert (node._spawn_x+node._odom_x, node._spawn_y+node._odom_y) == world
    assert node._odom_yaw == pytest.approx(.6)


@pytest.mark.parametrize('approaching', [True, False])
def test_surface_navigation_keeps_nav2_control_during_initial_turn(approaching):
    from thermal_motion_controller.runtime_policy import exploration_goal_due
    callback = method('thermal_motion_controller', 'controller_node', '_surface_timer',
        math=math, NAV2_DONE='done', STATE_COARSE_SURVEY='survey',
        exploration_goal_due=exploration_goal_due)
    sent = []
    node = NS(_sensor_model='a', _wx=0., _wy=0., _nav2_state='active',
        _surface_nav_goal=('src_1', 3., 0., 0.) if approaching else None,
        _surface_approach_timeout=20., _tracker_sources_t=5., _tracker_sources=[],
        _surface_wp=(3., 0.), _surface_plan_t=0., _explore_arrival=.6, _explore_timeout=45.,
        _explore_progress=NS(stalled=lambda *args:False),
        _send_nav2_goal=lambda x,y:sent.append((x,y)) or True,
        _nav2_progress_stalled=lambda *args:pytest.fail('Premature distance-only cancellation'),
        _pub=NS(publish=lambda cmd:pytest.fail('Direct control must not interrupt Nav2')))
    callback(node, 5.)  # 4-second legacy watchdog would cancel a normal heading turn.
    assert sent == [(3., 0.)]


def test_rpp_goal_tolerance_does_not_force_heading_only_control():
    import yaml
    config = yaml.safe_load((ROOT / 'thermal_bringup/config/nav2_params.yaml').read_text())
    controller = config['controller_server']['ros__parameters']
    tolerance = controller['general_goal_checker']['xy_goal_tolerance']
    pursuit = controller['FollowPath']
    # Humble RPP compares carrot distance with this goal tolerance. A larger
    # tolerance forces zero linear velocity even when the real goal is far away.
    assert 0. < tolerance < min(pursuit['min_lookahead_dist'], pursuit['lookahead_dist'])
    assert config['planner_server']['ros__parameters']['GridBased']['tolerance'] <= tolerance


def test_recorded_and_plotted_simulation_trajectory_preserves_world_position():
    scope = dict(math=math, np=np, SPAWN_X=-6., SPAWN_Y=0.)
    for filename, name in [('collect_sim_data.py', '_odom_cb'),
                           ('plot_all_figures.py', 'odom_to_world')]:
        path = ROOT / 'scripts' / filename
        function = next(n for n in ast.walk(ast.parse(path.read_text()))
                        if isinstance(n, ast.FunctionDef) and n.name == name)
        function.returns = None
        for arg in function.args.args:
            arg.annotation = None
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), scope)
    node = NS(_ts=lambda:1., _record_rate=lambda *args:None, _traj=[])
    message = NS(pose=NS(pose=NS(position=NS(x=-4., y=2.),
        orientation=NS(w=1., x=0., y=0., z=0.))),
        twist=NS(twist=NS(linear=NS(x=.2), angular=NS(z=0.))))
    scope['_odom_cb'](node, message)
    _, x, y, _ = scope['odom_to_world'](node._traj)
    np.testing.assert_array_equal([x[0], y[0]], [-4., 2.])
    # Historical CSVs without explicit world columns remain readable.
    _, x, y, _ = scope['odom_to_world']([dict(t=1., x=2., y=2., yaw=0.)])
    np.testing.assert_array_equal([x[0], y[0]], [-4., 2.])


def test_controller_information_gain_matches_analytic_localization_and_falls_back():
    callback = method('thermal_motion_controller', 'controller_node', '_posterior_gain',
                      np=np, source_information_gain=source_information_gain)
    source = NS(position=NS(x=.5, y=.5), existence_probability=1.,
                covariance_xx=.2, covariance_xy=0., covariance_yy=.2)
    message = NS(sources=[source])
    node = NS(_active_belief=lambda:message, _posterior_pd=1., _posterior_pf=0.,
              _posterior_variance=.2, _residual_planner_footprint_radius=1.)
    grid = dict(height=1, width=2, origin_x=0., origin_y=0., resolution=1.)
    # Certain detection/existence: only localization remains; det(I+I)=4.
    expected = np.log(2) * np.array([[1., np.exp(-.5)]])
    np.testing.assert_allclose(callback(node, grid), expected, atol=1e-10)
    message.sources.append(source)
    np.testing.assert_allclose(callback(node, grid), 2*expected, atol=1e-10)
    source.covariance_xx = float('nan')
    assert callback(node, grid) is None
    node._active_belief = lambda:None
    assert callback(node, grid) is None


@pytest.mark.parametrize('threshold,blocked', [(65, False), (40, True)])
def test_mapper_configured_threshold_changes_visibility(threshold, blocked):
    callback = method('thermal_field_reconstructor', 'thermal_mapper_node', '_slam_map_cb',
                      visibility=visibility, math=math)
    node = NS(_occupied_threshold=threshold, _pose_source='world', _spawn_x=0., _spawn_y=0.)
    origin = NS(position=NS(x=0., y=0.), orientation=NS(w=1., x=0., y=0., z=0.))
    message = NS(data=[0, 50, 0], info=NS(width=3, height=1, resolution=1., origin=origin))
    callback(node, message)
    assert bool(visibility.occupied_at(node._occ_view, [.5+1], [.5])[0]) == blocked
    assert visibility.line_reachable_known_free(node._occ_view, .5, .5, 2.5, .5) != blocked



@pytest.mark.parametrize('measurement_type', ['surface_radiance','field_direct'])
def test_slow_node_publishes_canonical_identity_and_historical_age(measurement_type):
    from thermal_motion_controller.belief import SourceBelief, BeliefParams
    from thermal_motion_controller.source_tracking import TrackedSource
    callback=method('thermal_motion_controller','belief_node','_tick',np=np,time=time,
        TrackedSource=TrackedSource, BeliefState=lambda:NS(sources=[],cardinality_pmf=[]),
        SourceEstimate=lambda:NS(position=NS(x=0.,y=0.)))
    header=NS(stamp=NS(sec=100,nanosec=0),frame_id='world')
    source=NS(id='canonical_42',position=NS(x=.5,y=.5),strength=20.,sigma=.6,
        existence_probability=.99,confidence=.95,observations=8,age_s=50.,
        covariance_xx=.2,covariance_xy=0.,covariance_yy=.2,status='confirmed')
    output=[]
    node=NS(latest=NS(header=header,sources=[source]),latest_map=NS(header=header,
        height=1,width=1,temperature_mean=[22.],confidence=[1.],last_seen_age_s=[0.],
        resolution=1.,origin_x=0.,origin_y=0.,measurement_type=measurement_type),
        mode='online',last_stamp=None,revision=0,ambient=22.,freshness=1.5,
        params=BeliefParams(budget_ms=1000),belief=SourceBelief(BeliefParams(budget_ms=1000)),
        pub=NS(publish=output.append),get_logger=lambda:NS(info=lambda text:None,error=pytest.fail))
    callback(node)
    result=output[-1].sources[0]
    assert result.id=='canonical_42' and result.status=='confirmed' and result.age_s==50.
    assert output[-1].health=='ready'
    callback(node)
    assert len(output)==1  # A cached registry frame is not another observation.


def test_surface_belief_does_not_require_gaussian_field_arrays():
    """Surface estimates remain usable even without A-only map payloads."""
    from thermal_motion_controller.belief import SourceBelief, BeliefParams
    from thermal_motion_controller.source_tracking import TrackedSource
    callback = method('thermal_motion_controller', 'belief_node', '_tick', np=np, time=time,
        TrackedSource=TrackedSource, BeliefState=lambda:NS(sources=[], cardinality_pmf=[]),
        SourceEstimate=lambda:NS(position=NS(x=0., y=0.)))
    output = []
    node = NS(latest=NS(header=NS(stamp=NS(sec=1, nanosec=0), frame_id='world'), sources=[]),
        latest_map=NS(measurement_type='surface_radiance'), mode='online', last_stamp=None,
        revision=0, params=BeliefParams(budget_ms=1000),
        belief=SourceBelief(BeliefParams(budget_ms=1000)), pub=NS(publish=output.append),
        get_logger=lambda:NS(info=lambda text:None, error=pytest.fail))
    callback(node)
    assert output[-1].health == 'ready'


@pytest.mark.parametrize('sensor,strategy', [('a','fast'), ('a','dual'), ('a','gp_ucb'), ('b','full')])
def test_static_control_runs_without_legacy_gradient_statistics(sensor, strategy):
    callback = method('thermal_motion_controller', 'controller_node', '_timer_cb', time=time)
    calls = []
    node = NS(_t0=time.monotonic(), _sensor_model=sensor, _strategy_mode=strategy,
        _update_world_pos_from_tf=lambda:False, _surface_timer=calls.append)
    # No gradient message/statistic helpers: modern navigation must still run.
    callback(node)
    assert len(calls) == 1


def test_slow_feedback_matches_by_id_even_when_another_source_is_nearer():
    from thermal_motion_controller.position_filter import PositionFilter
    from thermal_motion_controller.runtime_policy import slow_output_usable
    callback=method('thermal_motion_controller','source_tracker_node','_prior_cb',
                    np=np,slow_output_usable=slow_output_usable)
    tracks=[NS(track_id=key,x=x,y=0.,status='confirmed',last_seen_s=10.)
            for key,x in [('a',0.),('b',.3)]]
    filters={t.track_id:PositionFilter(np.array([t.x,0.]),np.eye(2)*.2) for t in tracks}
    node=NS(_slow_prior_enabled=True,_last_prior_revision=-1,_strategy='dual',_slow_timeout=3.,
        _last_update=10.,get_clock=lambda:NS(now=lambda:NS(nanoseconds=10_000_000_000)),
        _tracker=NS(estimator_model='kalman',tracks=tracks,_filters=filters,max_detection_age_s=1.5,gate_m=1.25))
    source=NS(id='b',status='confirmed',position=NS(x=.05,y=0.),covariance_xx=.2,covariance_yy=.2)
    message=NS(mode='online',health='ready',revision=1,header=NS(stamp=NS(sec=10,nanosec=0)),sources=[source])
    callback(node,message)
    assert filters['a'].state[0]==0. and .05<filters['b'].state[0]<.3
    before=filters['b'].state.copy()
    source.id='unknown';message.revision=2
    callback(node,message)
    np.testing.assert_array_equal(filters['b'].state,before)


def test_controller_retains_known_source_and_processed_id_across_observation_gap():
    callback=method('thermal_motion_controller','controller_node','_sources_cb',time=time)
    source=NS(id='src_1',status='confirmed',position=NS(x=2.,y=3.),strength=20.,
        existence_probability=.99,confidence=.9,observations=8,age_s=1000.,
        covariance_xx=.2,covariance_yy=.2,sigma=.6)
    node=NS(_strategy_mode='dual',_ambient_est=22.,_surface_seen_ids={'src_1'})
    callback(node,NS(sources=[source]))
    assert node._found_sources==[(2.,3.,42.)] and node._surface_seen_ids=={'src_1'}
    source.age_s=0.;source.observations+=1
    callback(node,NS(sources=[source]))
    assert node._surface_seen_ids=={'src_1'} and len(node._found_sources)==1
