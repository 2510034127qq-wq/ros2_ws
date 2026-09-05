"""Calibrated perspective RGB-D/thermal geometry, pure NumPy.

Optical frame: x right, y down, z forward. Robot/world: x forward,
y left, z up. Depth is optical z in metres, not Euclidean ray range.
"""
from dataclasses import dataclass
import math
import numpy as np
from .observation import ThermalObservation, MEASUREMENT_SURFACE_RADIANCE


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self):
        if self.width <= 0 or self.height <= 0 or self.fx <= 0 or self.fy <= 0:
            raise ValueError('invalid camera intrinsics')

    @classmethod
    def from_hfov(cls,width,height,hfov_deg):
        if not 0 < hfov_deg < 180:
            raise ValueError('hfov must be between 0 and 180 degrees')
        f=width/(2*math.tan(math.radians(hfov_deg)/2))
        return cls(width,height,f,f,(width-1)/2,(height-1)/2)

    def rays(self):
        v,u=np.indices((self.height,self.width))
        return np.stack(((u-self.cx)/self.fx,(v-self.cy)/self.fy,np.ones_like(u)),axis=-1)


@dataclass(frozen=True)
class SensorPose3D:
    x: float
    y: float
    z: float
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    frame_id: str = 'world'

    @property
    def rotation(self):
        cy,sy=math.cos(self.yaw),math.sin(self.yaw)
        cp,sp=math.cos(self.pitch),math.sin(self.pitch)
        cr,sr=math.cos(self.roll),math.sin(self.roll)
        rz=np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
        ry=np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
        rx=np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
        optical=np.array([[0,0,1],[-1,0,0],[0,-1,0]])
        return rz@ry@rx@optical

    @property
    def translation(self):
        return np.array([self.x,self.y,self.z])


class PerspectiveProjector:
    measurement_type=MEASUREMENT_SURFACE_RADIANCE

    def __init__(self,intrinsics,near_m=.15,far_m=15.):
        self.intrinsics=intrinsics
        self.near_m,self.far_m=near_m,far_m

    def project(self,image,depth,pose,stamp_s,confidence=None):
        image=np.asarray(image,dtype=float); depth=np.asarray(depth,dtype=float)
        shape=(self.intrinsics.height,self.intrinsics.width)
        if image.shape!=shape or depth.shape!=shape:
            raise ValueError('image, depth and calibrated dimensions must agree')
        conf=np.ones(shape) if confidence is None else np.asarray(confidence,dtype=float)
        if conf.shape!=shape:
            raise ValueError('confidence dimensions differ')
        valid=np.isfinite(image)&np.isfinite(depth)&(depth>=self.near_m)&(depth<=self.far_m)
        valid &= np.isfinite(conf)&(conf>0)
        optical=self.intrinsics.rays()[valid]*depth[valid,None]
        world=optical@pose.rotation.T+pose.translation
        return ThermalObservation(float(stamp_s),pose,self.measurement_type,
                                  world[:,0],world[:,1],image[valid],conf[valid],
                                  sample_wz=world[:,2],sample_range_m=np.linalg.norm(optical,axis=1))


def register_depth(depth,depth_intrinsics,thermal_intrinsics,depth_to_thermal):
    """Z-buffer registration using a calibrated 4x4 optical-frame transform.

    Holes stay NaN; no invented depth or nearest-neighbour hole filling.
    """
    depth=np.asarray(depth,dtype=float)
    transform=np.asarray(depth_to_thermal,dtype=float)
    if depth.shape!=(depth_intrinsics.height,depth_intrinsics.width) or transform.shape!=(4,4):
        raise ValueError('invalid depth calibration shapes')
    valid=np.isfinite(depth)&(depth>0)
    points=depth_intrinsics.rays()[valid]*depth[valid,None]
    points=points@transform[:3,:3].T+transform[:3,3]
    points=points[points[:,2]>0]
    k=thermal_intrinsics
    u=np.rint(k.fx*points[:,0]/points[:,2]+k.cx).astype(int)
    v=np.rint(k.fy*points[:,1]/points[:,2]+k.cy).astype(int)
    valid=(u>=0)&(u<k.width)&(v>=0)&(v<k.height)
    zbuf=np.full(k.width*k.height,np.inf)
    np.minimum.at(zbuf,v[valid]*k.width+u[valid],points[valid,2])
    zbuf[~np.isfinite(zbuf)]=np.nan
    return zbuf.reshape(k.height,k.width).astype(np.float32)


def radiometric_celsius(raw,scale_kelvin=.01,offset_kelvin=0.):
    """Convert configured TLinear units to C; raw DN needs its own calibration."""
    if scale_kelvin<=0 or not math.isfinite(scale_kelvin):
        raise ValueError('temperature scale must be positive')
    raw=np.asarray(raw)
    output=raw.astype(np.float32)*scale_kelvin+offset_kelvin-273.15
    output[(raw==0)|~np.isfinite(raw)]=np.nan
    return output
