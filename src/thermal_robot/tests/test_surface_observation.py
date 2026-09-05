"""B-level geometry/occlusion and radiometric input requirements."""
import sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
for pkg in ('thermal_sensor_sim','thermal_field_reconstructor'):
    sys.path.insert(0,str(ROOT/pkg))
from thermal_field_reconstructor.perspective import (
    CameraIntrinsics,SensorPose3D,PerspectiveProjector,register_depth,radiometric_celsius)
from thermal_sensor_sim.surface_scene import SurfaceObject,SurfaceRenderer,SensorEffects


def renderer(objects):
    k=CameraIntrinsics.from_hfov(65,49,57.)
    e=SensorEffects(noise_std_c=0,bias_drift_c_s=0,emissivity_std=0,depth_noise_std_m=0)
    return SurfaceRenderer(k,objects,effects=e)


def test_box_depth_and_front_surface_projection():
    r=renderer([SurfaceObject('heater',(4,0,.6),(1,1,1.2),temperature=60,emissivity=1)])
    pose=SensorPose3D(0,0,.6)
    im,depth=r.render(pose,0.)
    assert depth[24,32]==pytest.approx(3.5)
    assert im[24,32]==pytest.approx(60)
    obs=PerspectiveProjector(r.intrinsics).project(im,depth,pose,1.)
    hot=obs.temperature>50
    assert np.max(abs(obs.sample_wx[hot]-3.5))<1e-5
    assert obs.frame_id=='world' and obs.measurement_type=='surface_radiance'


def test_occluder_blocks_heat_and_back_face_is_not_visible():
    heater=SurfaceObject('heater',(4,0,.6),(1,1,1.2),temperature=22,
                         face_temperatures={1:80},emissivity=1)
    r=renderer([heater])
    im,_=r.render(SensorPose3D(0,0,.6),0.)
    assert im.max()==pytest.approx(22.)
    heater.face_temperatures={0:80}
    im,_=r.render(SensorPose3D(0,0,.6),0.)
    assert im.max()==pytest.approx(80.)
    r.objects.append(SurfaceObject('wall',(2,0,1),(1,3,2),temperature=22))
    im,depth=r.render(SensorPose3D(0,0,.6),0.)
    assert im[24,32]==pytest.approx(22.) and depth[24,32]==pytest.approx(1.5)


def test_cylinder_depth_and_camera_rotation():
    r=renderer([SurfaceObject('cylinder',(0,4,.6),(1,1,1.2),'cylinder',temperature=60,emissivity=1)])
    im,depth=r.render(SensorPose3D(0,0,.6,yaw=np.pi/2),0.)
    assert depth[24,32]==pytest.approx(3.5) and im[24,32]==pytest.approx(60.)


def test_depth_registration_and_invalid_radiometry():
    k=CameraIntrinsics.from_hfov(5,3,57.)
    depth=np.full((3,5),2.)
    assert np.allclose(register_depth(depth,k,k,np.eye(4)),depth)
    depth[1,2]=np.nan
    assert np.isnan(register_depth(depth,k,k,np.eye(4))[1,2])
    values=radiometric_celsius(np.array([0,30000],dtype=np.uint16))
    assert np.isnan(values[0]) and values[1]==pytest.approx(26.85,abs=.001)


def test_emissivity_bias_and_drift_have_physical_direction():
    r=renderer([SurfaceObject('heater',(4,0,.6),(1,1,1.2),temperature=60,emissivity=.5)])
    im,_=r.render(SensorPose3D(0,0,.6),0.)
    assert 22<im[24,32]<60
    r.effects.bias_drift_c_s=.1
    later,_=r.render(SensorPose3D(0,0,.6),10.)
    assert later[24,32]-im[24,32]==pytest.approx(1.)


def test_no_depth_does_not_create_map_evidence():
    k=CameraIntrinsics.from_hfov(5,3,57.)
    obs=PerspectiveProjector(k).project(np.ones((3,5))*40,np.full((3,5),np.nan),SensorPose3D(0,0,1),0.)
    assert len(obs.temperature)==0


def test_dynamic_fusion_does_not_keep_an_extinguished_source_hot():
    from thermal_field_reconstructor.thermal_mapping import WorldThermalGrid
    from thermal_field_reconstructor.observation import ThermalObservation,SensorPose2D
    g=WorldThermalGrid(center_x=0,center_y=0,size_x_m=4,size_y_m=4,resolution=1,fusion_memory_s=1.)
    pose=SensorPose2D(0,0,0)
    for t in range(30):
        g.integrate_observation(ThermalObservation(float(t),pose,'field_direct',np.array([.5]),np.array([.5]),
            np.array([60. if t<10 else 22.]),np.array([1.])))
    assert g.snapshot(29).temperature_mean[2,2]<22.1


def test_surface_world_contains_collision_bodies_and_state_plugin(tmp_path):
    from thermal_sensor_sim.surface_scene import write_surface_world,objects_from_world
    from thermal_sensor_sim.scenario import load_scenario_file
    import xml.etree.ElementTree as ET
    scenario=load_scenario_file(ROOT/'thermal_bringup/config/scenarios/static_two_sources.yaml')
    output=tmp_path/'b.world'
    write_surface_world(ROOT/'thermal_bringup/worlds/thermal_scene_nav.world',scenario,output)
    world=ET.parse(output).getroot().find('world')
    names={m.get('name') for m in world.findall('model')}
    assert 'thermal_body_T1_north' in names and 'thermal_body_T2_east' in names
    assert world.find("plugin[@name='thermal_surface_state']") is not None
    assert len(objects_from_world(output))>=2


def test_rotated_occupancy_preserves_obstacle_and_motion_queries():
    from thermal_field_reconstructor import visibility as v
    data=np.zeros((4,4),dtype=np.int16);data[1,2]=100
    view=v.OccupancyView(0,0,1,data)
    rotated=v.transform_view(view,10,20,np.pi/2)
    assert v.occupied_at(rotated,np.array([8.5]),np.array([22.5]))[0]
    assert v.known_free_at(rotated,np.array([8.5]),np.array([20.5]))[0]
    assert not v.line_reachable_known_free(rotated,8.5,20.5,8.5,23.5)
    assert v.line_reachable_known_free(rotated,9.5,20.5,9.5,23.5)


def test_fractional_confidence_weights_mean_and_expires_sector_history():
    from thermal_field_reconstructor.thermal_mapping import WorldThermalGrid
    from thermal_field_reconstructor.observation import ThermalObservation,SensorPose2D
    g=WorldThermalGrid(center_x=0,center_y=0,size_x_m=4,size_y_m=4,
                       resolution=1,sector_memory_s=5.)
    def observe(t,temp,weight):
        g.integrate_observation(ThermalObservation(t,SensorPose2D(-1.5,.5,0),
            'surface_radiance',np.array([.5]),np.array([.5]),np.array([temp]),np.array([weight])))
    observe(0.,60.,.1)
    assert g.snapshot(0).temperature_mean[2,2]==pytest.approx(60.)
    observe(1.,20.,.3)
    snap=g.snapshot(1.)
    assert snap.temperature_mean[2,2]==pytest.approx(30.)
    assert snap.confidence[2,2]==pytest.approx(.4/6)
    assert snap.last_view_distance_m[2,2]==pytest.approx(2.)
    assert snap.view_sectors[2,2]>0
    assert g.snapshot(7.).view_sectors[2,2]==0
