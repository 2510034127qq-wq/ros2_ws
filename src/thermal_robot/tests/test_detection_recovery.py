"""Regressions for identity theft, negative evidence, and incomplete exploration."""
import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'thermal_motion_controller'))
from thermal_motion_controller.motion_filter import MotionFilter, associate
from thermal_motion_controller.source_tracking import SourceTrackerCore, SourceDetection
from thermal_motion_controller.runtime_policy import CoverageSweep, exploration_goal_due
from thermal_motion_controller.revisit import RevisitScheduler


def detection(x, y=0.):
    return SourceDetection(x, y, 25., .95, .35)


def test_uncertain_lost_track_cannot_steal_a_precise_tracks_observation():
    lost = MotionFilter(np.array([4., 0., 0., 0.]), np.eye(4)*100.)
    current = MotionFilter(np.array([4.1, 0., 0., 0.]), np.eye(4)*.1)
    matches, unmatched = associate([lost, current], [detection(4.)])
    assert matches == [(1, 0)] and not unmatched


def test_long_unobserved_drift_cannot_relabel_a_different_heater():
    tracker = SourceTrackerCore(motion_model='kalman', gate_m=3., confirm_observations=3)
    for t in range(10):
        tracker.update([detection(.2*t)], float(t))
    old = tracker.tracks[0]
    for t in range(10, 40):
        tracker.update([], float(t))
    assert old.x < 3.5 and abs(old.vx) < .001
    tracker.update([detection(8.)], 40.)
    assert len(tracker.tracks) == 2
    assert old.last_seen_s == 9.
    assert tracker.tracks[1].track_id != old.track_id


def confirmed_tracker():
    tracker = SourceTrackerCore(motion_model='kalman', max_detection_age_s=1.5,
                                confirm_observations=3)
    for t in range(8):
        tracker.update([detection(2.125, 2.125)], float(t))
    assert tracker.tracks[0].status == 'confirmed'
    return tracker


@pytest.mark.parametrize('visibility', ['cold', 'unseen', 'old', 'hot_neighbour'])
def test_only_current_cold_evidence_retires_a_heat_source(visibility):
    tracker = confirmed_tracker()
    temp = np.full((20, 20), 22.)
    conf = np.ones_like(temp)
    age = np.zeros_like(temp)
    if visibility == 'unseen':
        conf[:] = 0.
    elif visibility == 'old':
        age[:] = 20.
    elif visibility == 'hot_neighbour':
        # Partial hot footprint at an observation boundary isn't a source peak,
        # but still prevents negative evidence against the known source.
        temp[8, 9] = 50.
        conf[8, 10] = 0.
    for t in range(8, 14):
        tracker.update_from_map(temp, conf, .25, 0., 0., float(t), age)
    track = tracker.tracks[0]
    if visibility == 'cold':
        assert track.status == 'stale' and track.existence_probability < .1
        tracker.update([detection(2.125, 2.125)], 14.)
        assert track.status == 'candidate'  # A single warm return isn't confirmation.
    else:
        assert track.existence_probability > .7


def test_repeated_cached_cold_frame_is_not_independent_death_evidence():
    tracker = confirmed_tracker()
    temp = np.full((20, 20), 22.)
    for now in (8., 8.5, 9., 9.5):
        tracker.update_from_map(temp, np.ones_like(temp), .25, 0., 0., now,
                                np.full_like(temp, now-8.))
    assert tracker.tracks[0].existence_probability > .85


@pytest.mark.parametrize('occluded', [False, True])
def test_surface_render_projection_fusion_and_tracker_distinguish_extinction_from_occlusion(occluded):
    for package in ('thermal_sensor_sim', 'thermal_field_reconstructor'):
        sys.path.insert(0, str(ROOT/package))
    from thermal_sensor_sim.surface_scene import SurfaceObject, SurfaceRenderer, SensorEffects
    from thermal_field_reconstructor.perspective import CameraIntrinsics, SensorPose3D, PerspectiveProjector
    from thermal_field_reconstructor.thermal_mapping import WorldThermalGrid
    camera = CameraIntrinsics.from_hfov(65,49,57.)
    heater = SurfaceObject('heater',(4.,0.,.6),(1.,1.,1.2),temperature=60.,emissivity=1.)
    renderer = SurfaceRenderer(camera,[heater],effects=SensorEffects(
        noise_std_c=0.,bias_drift_c_s=0.,emissivity_std=0.,depth_noise_std_m=0.))
    projector = PerspectiveProjector(camera)
    pose = SensorPose3D(0.,0.,.6)
    grid = WorldThermalGrid(center_x=4.,center_y=0.,size_x_m=10.,size_y_m=10.,
                            resolution=.25,fusion_memory_s=3.)
    tracker = SourceTrackerCore(motion_model='kalman',max_detection_age_s=1.5,confirm_observations=3)
    tracker.measurement_type = 'surface_radiance'
    for step in range(41):
        now = step*.5
        if step == 12:
            assert tracker.tracks[0].status == 'confirmed'
            if occluded:
                renderer.objects.append(SurfaceObject('wall',(2.,0.,1.),(1.,3.,2.),temperature=22.))
            else:
                heater.temperature = 22.
        thermal, depth = renderer.render(pose,now)
        grid.integrate_observation(projector.project(thermal,depth,pose,now))
        snapshot = grid.snapshot(now)
        tracker.update_from_map(snapshot.temperature_mean,snapshot.confidence,.25,
            snapshot.origin_x,snapshot.origin_y,now,snapshot.last_seen_age_s)
    assert len(tracker.tracks) == 1
    track = tracker.tracks[0]
    if occluded:
        assert track.existence_probability > .45
    else:
        assert track.status == 'stale' and track.existence_probability < .1


def test_sweep_requires_observed_rotation_and_restarts_after_travel():
    sweep = CoverageSweep(distance_m=4., interval_s=60.)
    assert sweep.step((0., 0.), 0., 0.)
    assert sweep.step((0., 0.), 0., 30.)  # Commanded rotation without odometry doesn't count.
    for i, yaw in enumerate(np.linspace(.1, 2*math.pi, 64)):
        active = sweep.step((0., 0.), math.atan2(math.sin(yaw), math.cos(yaw)), 30.+i*.1)
    assert not active
    assert not sweep.step((1., 0.), 0., 40.)
    assert sweep.step((4.1, 0.), 0., 41.)


def test_exploration_does_not_abandon_goal_on_old_eight_second_timer():
    assert not exploration_goal_due((4., 0.), (1.5, 0.), 9., 0.)
    assert not exploration_goal_due((4., 0.), (3., 0.), 15., 0.)
    assert exploration_goal_due((4., 0.), (3.5, 0.), 20., 0.)
    assert exploration_goal_due((4., 0.), (1., 0.), 46., 0.)


def test_stuck_exploration_avoids_failed_goal_until_cooldown():
    from thermal_motion_controller.runtime_policy import ExplorationProgress
    progress = ExplorationProgress(timeout_s=12., cooldown_s=45., exclusion_m=2.)
    assert not progress.stalled((0.,0.), 0.)
    assert not progress.stalled((1.,0.), 10.)
    assert not progress.stalled((1.,0.), 21.)
    assert progress.stalled((1.,0.), 22.)
    progress.reject((4.,0.), 22.)
    assert progress.allowed(np.array([4.,8.]), np.zeros(2), 23.).tolist() == [False,True]
    assert progress.allowed(4.,0.,68.)


def test_equal_unknown_regions_prefer_less_turning_but_heat_can_override():
    from thermal_motion_controller.planning import select_residual_target
    empty = np.zeros((41,41))
    args = (0.,0.,41,41,.25,-5.125,-5.125)
    target = select_residual_target(*args,empty,empty,empty,min_d=2.,max_d=4.,
                                   heading_yaw=0.,w_turn=.25)
    assert target.x > 0. and abs(target.y) < .5
    hot = empty.copy()
    hot[18:23,6:11] = 20.
    target = select_residual_target(*args,hot,empty,empty,min_d=2.,max_d=4.,
                                   heading_yaw=0.,w_turn=.25)
    assert target.x < 0.


def test_disproved_source_cannot_monopolize_revisit_with_large_covariance():
    scheduler = RevisitScheduler()
    source = dict(id='old', x=5., y=0., age_s=15., probability=.05, strength=25.,
                  status='stale', covariance_xx=10000., covariance_yy=10000.)
    assert scheduler.select([source], 20., (0., 0.)) is None


def test_evaluation_counts_physical_sources_and_identity_switches_separately():
    sys.path.insert(0, str(ROOT/'scripts'))
    from evaluate_detection_tracks import evaluate
    truth = [dict(t=float(t), sources=[
        dict(id='A', x=0., y=0., status='truth_active'),
        dict(id='B', x=5., y=0., status='truth_active'),
        dict(id='C', x=10., y=0., status='truth_active')]) for t in range(8)]
    rows = [dict(t=float(t), id='same_id', status='confirmed', age=0., x=0. if t<4 else 5., y=0.)
            for t in range(8)]
    result = evaluate(rows, truth)
    assert result['window_recall'] == pytest.approx(2/3)
    assert result['identity_switch_count'] == 1
    assert result['missed_truth_ids'] == ['C']
