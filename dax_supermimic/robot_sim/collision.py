"""Compiled-geometry collision queries shared by generation and probes."""
import xml.etree.ElementTree as ET
import numpy as np
import mujoco

class CollisionChecker:
    """Pair distances and conservative joint-motion bounds for piecewise linear q paths."""
    def __init__(self,path,joint_names):
        self.m=mujoco.MjModel.from_xml_path(str(path));m=self.m
        self.d=mujoco.MjData(m)
        self.ids=np.array([m.joint(n).qposadr[0] for n in joint_names])
        self.jids=[m.joint(n).id for n in joint_names]
        if m.nq!=len(joint_names) or any(m.jnt_type[j]!=mujoco.mjtJoint.mjJNT_HINGE for j in self.jids):
            raise ValueError('Certificate supports exactly the named revolute18 model')
        if any(np.linalg.norm(m.jnt_pos[j])>1e-10 for j in self.jids):
            raise ValueError('Revolute pivots must coincide with body origins')
        self.geoms=np.flatnonzero((m.geom_contype!=0)|(m.geom_conaffinity!=0))
        exclusions=[];excluded=set()
        for x in ET.parse(path).getroot().findall('.//exclude'):
            a,b=m.body(x.get('body1')).id,m.body(x.get('body2')).id
            excluded.add(tuple(sorted((a,b))));exclusions.append([x.get('body1'),x.get('body2')])
        pairs=[];suppressed=[]
        for ix,a in enumerate(self.geoms):
            for b in self.geoms[ix+1:]:
                ba,bb=int(m.geom_bodyid[a]),int(m.geom_bodyid[b])
                reason='same_body' if ba==bb else 'explicit_exclusion' if tuple(sorted((ba,bb))) in excluded else None
                if reason:suppressed.append({'geoms':[m.geom(int(a)).name,m.geom(int(b)).name],'reason':reason})
                else:pairs.append((int(a),int(b)))
        self.a=np.array([p[0]for p in pairs]);self.b=np.array([p[1]for p in pairs])
        # Bound distance of every point on a geom to each ancestor revolute pivot:
        # sum of fixed body translations + geom-center translation + enclosing geom radius.
        ancestors=np.zeros((m.ngeom,len(self.ids)),bool);lever=np.zeros_like(ancestors,dtype=float)
        bybody={int(m.jnt_bodyid[j]):i for i,j in enumerate(self.jids)}
        for g in self.geoms:
            body=int(m.geom_bodyid[g]);radius=float(np.linalg.norm(m.geom_pos[g])+m.geom_rbound[g])
            while body:
                if body in bybody:
                    idx=bybody[body];ancestors[g,idx]=True;lever[g,idx]=radius
                radius+=float(np.linalg.norm(m.body_pos[body]));body=int(m.body_parentid[body])
        # Shared ancestor rotations preserve pair distances exactly, so remove them.
        shared=ancestors[self.a]&ancestors[self.b]
        self.bounds=np.where(shared,0.,lever[self.a]+lever[self.b])
        self.metadata={'collision_geoms':[m.geom(int(g)).name for g in self.geoms],
            'checked_pairs':[[m.geom(int(a)).name,m.geom(int(b)).name]for a,b in pairs],
            'explicit_body_exclusions':exclusions,'suppressed_pairs':suppressed,
            'semantics':'MuJoCo compiled convex mesh distances; no contact-mask or implicit parent filtering beyond named exclusions and same body',
            'hard_rule':'any signed distance <= 1e-6m fails admission; 1e-6 is numerical guard, not hardware safety margin',
            'certificate':'midpoint pair distance > sum(nonshared ancestor lever*half angular interval)+guard; recursively subdivide otherwise',
            'limit':'conditional model-geometric certificate for linear interpolation; not original nonconvex CAD, physical tracking or hardware safety'}
        self.calls=0

    def fk(self,q):
        self.d.qpos[self.ids]=q;mujoco.mj_kinematics(self.m,self.d)

    def distances(self,q):
        self.fk(q);d=self.d;m=self.m
        center=np.linalg.norm(d.geom_xpos[self.a]-d.geom_xpos[self.b],axis=1)
        lower=center-m.geom_rbound[self.a]-m.geom_rbound[self.b]
        # Positive bounding-sphere gaps are conservative distance lower bounds.
        values=lower.copy()
        for i in np.flatnonzero(lower<=0):
            values[i]=mujoco.mj_geomDistance(m,d,int(self.a[i]),int(self.b[i]),10.,None)
        self.calls+=1
        return values

    def posture(self,q):
        ds=self.distances(q);idx=int(ds.argmin())
        return {'passed':bool(ds[idx]>1e-6),'min_distance_lower_bound_m':float(ds[idx]),
                'limiting_pair':self.metadata['checked_pairs'][idx]}

    def path(self,q0,q1,max_depth=12,max_nodes=2047):
        nodes=0;minimum=float('inf');max_used=0
        for q in (q0,q1):
            d=self.distances(q);minimum=min(minimum,float(d.min()))
            if d.min()<=1e-6:return {'status':'rejected_collision','nodes':nodes,'min_distance_lower_bound_m':minimum}
        stack=[(np.asarray(q0),np.asarray(q1),0)]
        while stack:
            a,b,depth=stack.pop();nodes+=1;max_used=max(max_used,depth)
            if nodes>max_nodes:return {'status':'unknown_budget','nodes':nodes,'max_depth':max_used}
            ds=self.distances((a+b)/2);minimum=min(minimum,float(ds.min()))
            if ds.min()<=1e-6:return {'status':'rejected_collision','nodes':nodes,'min_distance_lower_bound_m':minimum}
            motion=self.bounds@(np.abs(b-a)/2)
            if np.all(ds>motion+1e-6):continue
            if depth>=max_depth:return {'status':'unknown_resolution','nodes':nodes,'max_depth':max_used}
            mid=(a+b)/2;stack.extend([(a,mid,depth+1),(mid,b,depth+1)])
        return {'status':'certified_model_path','nodes':nodes,'max_depth':max_used,
                'min_sampled_distance_lower_bound_m':minimum}
