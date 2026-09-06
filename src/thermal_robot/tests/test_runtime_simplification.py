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


@pytest.mark.parametrize('measurement_type,expected_extractions',
                         [('surface_radiance', 1), ('field_direct', 2)])
def test_slow_births_skip_gaussian_prediction_only_for_surface(measurement_type, expected_extractions):
    extractions, predictions, births = [], [], []
    def extract(temp, *args):
        result = [float(temp[0, 0])]
        extractions.append(result)
        return result
    def predict(*args):
        predictions.append(args)
        return np.array([[25.]])
    def update(detections, stamp, visible, birth_candidates, snapshot):
        births.extend(birth_candidates)
        return False
    callback = method('thermal_motion_controller', 'belief_node', '_tick', np=np, time=time,
                      predict_field=predict, BeliefState=lambda:NS(sources=[], cardinality_pmf=[]))
    message = NS(header=NS(stamp=NS(sec=1, nanosec=0)), height=1, width=1, resolution=1.,
                 origin_x=0., origin_y=0., temperature_mean=[30.], confidence=[1.],
                 last_seen_age_s=[0.], measurement_type=measurement_type)
    output = []
    node = NS(latest=message, mode='online', last_stamp=None, ambient=22., freshness=1.5,
              revision=0, params=NS(budget_ms=1000), extractor=NS(extract_detections=extract),
              belief=NS(clusters=[], update=update, health='ready'), pub=NS(publish=output.append),
              get_logger=lambda:NS(info=lambda text:None, error=pytest.fail))
    callback(node)
    assert len(extractions) == expected_extractions
    assert len(predictions) == expected_extractions-1
    assert births == ([30.] if measurement_type == 'surface_radiance' else [27.])
    assert output[0].health == 'ready'
