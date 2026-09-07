"""B-level analytic 3D thermal surface renderer, without ROS or GPU.

Nearest intersections give self/scene occlusion and optical depth. Specified
surface temperatures are apparent radiometric samples with a grey-body T^4
mixing approximation. This is not a heat-transfer or spectral LWIR simulator.
"""
from dataclasses import dataclass, field
import math
import xml.etree.ElementTree as ET
import numpy as np


@dataclass
class SurfaceObject:
    name: str
    center: tuple
    size: tuple = (1.,1.,1.)
    shape: str = 'box'
    yaw: float = 0.
    temperature: float = 22.
    emissivity: float = .95
    # Box order: -x,+x,-y,+y,-z,+z. Cylinder: side,bottom,top.
    face_temperatures: dict = field(default_factory=dict)

    def intersect(self,origin,directions):
        c,s=math.cos(self.yaw),math.sin(self.yaw)
        rot=np.array([[c,-s,0],[s,c,0],[0,0,1]])
        o=(np.asarray(origin)-self.center)@rot
        d=directions@rot
        n=len(d)
        if self.shape=='box':
            half=np.asarray(self.size)/2
            parallel=np.abs(d)<1e-12
            safe=np.where(parallel,1.,d)
            a=(-half-o)/safe; b=(half-o)/safe
            low=np.minimum(a,b); high=np.maximum(a,b)
            low=np.where(parallel,-np.inf,low); high=np.where(parallel,np.inf,high)
            outside=np.any(parallel&(np.abs(o)>half),axis=1)
            enter=np.max(low,axis=1); leave=np.min(high,axis=1)
            inside=enter<0
            t=np.where(inside,leave,enter)
            axis=np.where(inside,np.argmin(high,axis=1),np.argmax(low,axis=1))
            sign=np.where(inside,d[np.arange(n),axis]>0,d[np.arange(n),axis]<0)
            face=2*axis+sign.astype(int)
            valid=(leave>=np.maximum(enter,0))&(t>1e-6)&~outside
            return np.where(valid,t,np.inf),face
        if self.shape!='cylinder':
            raise ValueError('surface shape must be box or cylinder')
        r=float(self.size[0])/2; half=float(self.size[2])/2
        a=np.sum(d[:,:2]**2,axis=1); b=2*(d[:,:2]@o[:2]); cc=o[0]**2+o[1]**2-r*r
        disc=b*b-4*a*cc
        root=np.sqrt(np.maximum(disc,0))
        with np.errstate(divide='ignore',invalid='ignore'):
            t1=(-b-root)/(2*a); t2=(-b+root)/(2*a)
        t=np.full(n,np.inf); face=np.zeros(n,dtype=int)
        for candidate in (t1,t2):
            z=o[2]+candidate*d[:,2]
            ok=(disc>=0)&(a>1e-12)&(candidate>1e-6)&(np.abs(z)<=half)&(candidate<t)
            t[ok]=candidate[ok]
        for z,k in ((-half,1),(half,2)):
            with np.errstate(divide='ignore',invalid='ignore'): cap=(z-o[2])/d[:,2]
            xy=o[:2]+cap[:,None]*d[:,:2]
            ok=(cap>1e-6)&(np.sum(xy*xy,axis=1)<=r*r)&(cap<t)&(np.abs(d[:,2])>1e-12)
            t[ok]=cap[ok];face[ok]=k
        return t,face


@dataclass
class SensorEffects:
    noise_std_c: float = .15
    bias_c: float = 0.
    bias_drift_c_s: float = .001
    emissivity_std: float = .01
    depth_noise_std_m: float = .01
    dropout_probability: float = 0.
    seed: int = 0


class SurfaceRenderer:
    def __init__(self,intrinsics,objects=(),ambient_c=22.,far_m=15.,effects=None):
        self.intrinsics=intrinsics
        self.objects=list(objects)
        self.ambient_c=ambient_c
        self.far_m=far_m
        self.effects=effects or SensorEffects()
        self.rng=np.random.default_rng(self.effects.seed)

    def render(self,pose,stamp_s,extra_objects=()):
        shape=(self.intrinsics.height,self.intrinsics.width)
        rays=self.intrinsics.rays().reshape(-1,3)@pose.rotation.T
        origin=pose.translation
        depth=np.full(len(rays),np.inf)
        temp=np.full(len(rays),self.ambient_c)
        emissivity=np.ones(len(rays))
        # Ground plane is opaque and has ambient surface temperature.
        with np.errstate(divide='ignore',invalid='ignore'): ground=-origin[2]/rays[:,2]
        ground_ok=(ground>0)&(ground<=self.far_m)
        depth[ground_ok]=ground[ground_ok]
        for obj in [*self.objects,*extra_objects]:
            t,face=obj.intersect(origin,rays)
            use=(t<depth)&(t<=self.far_m)
            surface=np.full(len(t),obj.temperature)
            for face_id,value in obj.face_temperatures.items():
                surface[face==int(face_id)]=value
            temp[use]=surface[use];depth[use]=t[use];emissivity[use]=obj.emissivity
        valid=np.isfinite(depth)
        e=self.effects
        emissivity=np.clip(emissivity+self.rng.normal(0,e.emissivity_std,len(depth)),.01,1)
        temp=(emissivity*(temp+273.15)**4+(1-emissivity)*(self.ambient_c+273.15)**4)**.25-273.15
        temp+=e.bias_c+e.bias_drift_c_s*stamp_s+self.rng.normal(0,e.noise_std_c,len(depth))
        depth+=self.rng.normal(0,e.depth_noise_std_m,len(depth))
        valid &= self.rng.random(len(depth))>=e.dropout_probability
        # Sky/far/no-return pixels carry no geometric observation.
        temp[~valid]=self.ambient_c;depth[~valid]=np.nan
        return temp.reshape(shape).astype(np.float32),depth.reshape(shape).astype(np.float32)


def objects_from_world(path,ambient_c=22.):
    """Parse static SDF box/cylinder collision geometry, including link poses."""
    def pose(element):
        values=[float(x) for x in element.findtext('pose','0 0 0 0 0 0').split()]
        return values
    result=[]
    root=ET.parse(path).getroot()
    for model in root.findall('.//world/model'):
        mp=pose(model)
        for link in model.findall('link'):
            lp=pose(link)
            for collision in link.findall('collision'):
                cp=pose(collision)
                # Supplied worlds have upright collision primitives. Reject
                # unsupported tilt rather than silently rendering bad geometry.
                if any(abs(v)>1e-6 for v in (mp[3],mp[4],lp[3],lp[4],cp[3],cp[4])):
                    raise ValueError('tilted SDF collision requires mesh renderer')
                def rotate(xy,yaw):
                    c,s=math.cos(yaw),math.sin(yaw)
                    return np.array([[c,-s],[s,c]])@xy
                xy=np.array(mp[:2])+rotate(lp[:2],mp[5])+rotate(cp[:2],mp[5]+lp[5])
                center=(*xy,mp[2]+lp[2]+cp[2])
                box=collision.find('geometry/box/size');cyl=collision.find('geometry/cylinder')
                if box is not None:
                    size=tuple(float(v) for v in box.text.split());shape='box'
                elif cyl is not None:
                    radius=float(cyl.findtext('radius'));height=float(cyl.findtext('length'))
                    size=(2*radius,2*radius,height);shape='cylinder'
                else:
                    continue
                result.append(SurfaceObject(model.get('name','object'),center,size,shape,
                                            mp[5]+lp[5]+cp[5],ambient_c))
    return result


def objects_from_sources(states,height=.8,diameter=.3,emissivity=.95,shape='box',hot_faces=(),ambient_c=22.):
    # Only the simulator calls this adapter. Published images/depth contain no IDs.
    objects=[]
    for state in states:
        amplitude=state.amplitude if state.active else 0.
        temperatures={int(face):ambient_c+amplitude for face in hot_faces}
        objects.append(SurfaceObject(state.source_id,(state.x,state.y,height/2),
                       (diameter,diameter,height),shape,temperature=ambient_c if hot_faces else ambient_c+amplitude,
                       emissivity=emissivity,face_temperatures=temperatures))
    return objects


def write_surface_world(world_path,scenario,output_path,height=.8,diameter=.3,shape='box'):
    """Add real Gazebo collision bodies paired with the analytic thermal surfaces."""
    tree=ET.parse(world_path);world=tree.getroot().find('world')
    if world is None: raise ValueError('missing SDF world')
    plugin=ET.SubElement(world,'plugin',name='thermal_surface_state',filename='libgazebo_ros_state.so')
    ros=ET.SubElement(plugin,'ros');ET.SubElement(ros,'namespace').text='/thermal_scene'
    ET.SubElement(plugin,'update_rate').text='10.0'
    for state in scenario.all_states(0.):
        model=ET.SubElement(world,'model',name='thermal_body_'+state.source_id)
        ET.SubElement(model,'static').text='true'
        ET.SubElement(model,'pose').text=f'{state.x} {state.y} {height/2} 0 0 0'
        link=ET.SubElement(model,'link',name='body')
        for kind in ('collision','visual'):
            element=ET.SubElement(link,kind,name=kind)
            geometry=ET.SubElement(element,'geometry')
            if shape=='box':
                box=ET.SubElement(geometry,'box');ET.SubElement(box,'size').text=f'{diameter} {diameter} {height}'
            elif shape=='cylinder':
                cylinder=ET.SubElement(geometry,'cylinder')
                ET.SubElement(cylinder,'radius').text=str(diameter/2)
                ET.SubElement(cylinder,'length').text=str(height)
            else: raise ValueError('unsupported surface shape')
            if kind=='visual':
                material=ET.SubElement(element,'material');ET.SubElement(material,'diffuse').text='0.8 0.25 0.1 1'
    tree.write(output_path,encoding='utf-8',xml_declaration=True)
    return str(output_path)
