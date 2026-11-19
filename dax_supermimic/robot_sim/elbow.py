"""Elbow geometry and legacy interval checks; no experiment orchestration."""
import numpy as np

def interval(lo,hi=None):
    lo=np.asarray(lo,dtype=float);hi=lo if hi is None else np.asarray(hi,dtype=float)
    return (np.nextafter(lo,-np.inf),np.nextafter(hi,np.inf))


def add(a,b):return interval(a[0]+b[0],a[1]+b[1])


def sub(a,b):return interval(a[0]-b[1],a[1]-b[0])


def mul(a,b):
    values=np.stack((a[0]*b[0],a[0]*b[1],a[1]*b[0],a[1]*b[1]))
    return interval(values.min(axis=0),values.max(axis=0))


def square(a):
    return interval(np.where((a[0]<=0)&(a[1]>=0),0,np.minimum(a[0]**2,a[1]**2)),np.maximum(a[0]**2,a[1]**2))


def total(a):return interval(np.sum(a[0]),np.sum(a[1]))


def dot(a,b):return total(mul(a,b))


def cross(a,b):
    lo=[];hi=[]
    for i,j,k in ((0,1,2),(1,2,0),(2,0,1)):
        v=sub(mul((a[0][j],a[1][j]),(b[0][k],b[1][k])),mul((a[0][k],a[1][k]),(b[0][j],b[1][j])))
        lo.append(v[0]);hi.append(v[1])
    return np.array(lo),np.array(hi)


class ElbowConstraint:
    """Shoulder--wrist-axis swivel sign and non-degeneracy, checked on both arms."""
    score_guard=1e-6
    axis_guard=1e-6
    bend_guard=1e-5
    gravity_projection_guard=1e-6

    def __init__(self,checker):
        self.checker=checker;m=checker.m
        self.bodies=np.array([[m.body(side+'_arm_link'+str(k)).id for k in (0,3,6)]for side in ('left','right')])
        # Bound motion of shoulder/elbow/wrist body origins under every ancestor joint.
        self.levers=np.zeros((2,3,len(checker.ids)))
        bybody={int(m.jnt_bodyid[j]):i for i,j in enumerate(checker.jids)}
        for arm in range(2):
            for k,b in enumerate(self.bodies[arm]):
                radius=0.;body=int(b)
                while body:
                    if body in bybody:self.levers[arm,k,bybody[body]]=radius
                    radius+=float(np.linalg.norm(m.body_pos[body]));body=int(m.body_parentid[body])
        self.metadata={'shoulder':['left_arm_link0','right_arm_link0'],'elbow':['left_arm_link3','right_arm_link3'],
             'wrist':['left_arm_link6','right_arm_link6'],'frame':'base_link; fixed root; gravity down [0,0,-1]',
             'elbow_anchor_type':'soft position preference seeded from elbow-down atlas; NOT an exact external elbow target',
             'hard_branch':'both arms projected elbow displacement dot projected down direction > score_guard',
             'score_guard_m':self.score_guard,'axis_guard_m':self.axis_guard,'bend_guard_m':self.bend_guard,
             'gravity_projection_guard':self.gravity_projection_guard,
             'degenerate_policy':'reject/unknown for straight arm or near-vertical shoulder-wrist axis',
             'path_certificate':'interval lower bound on elbow-down numerator plus non-degeneracy using body-origin motion bounds; recursively split linear q path',
             'scope':'probe-only numeric guards, not task clearance/safety limits or full elbow-angle cone preference'}

    @staticmethod
    def metrics(points):
        s,e,w=np.asarray(points);v=w-s;axis=np.linalg.norm(v)
        if axis<=ElbowConstraint.axis_guard:return {'passed':False,'status':'degenerate_axis','score_m':None}
        u=v/axis;g=np.array([0.,0.,-1.]);gp=g-np.dot(g,u)*u
        ep=e-s-np.dot(e-s,u)*u;gn=np.linalg.norm(gp);bend=np.linalg.norm(ep)
        if gn<=ElbowConstraint.gravity_projection_guard or bend<=ElbowConstraint.bend_guard:
            return {'passed':False,'status':'degenerate_swivel','score_m':None,'bend_m':float(bend),'gravity_projection':float(gn)}
        score=float(ep@gp/gn)
        return {'passed':score>ElbowConstraint.score_guard,'status':'elbow_down'if score>ElbowConstraint.score_guard else'elbow_up_or_boundary',
                'score_m':score,'bend_m':float(bend),'axis_length_m':float(axis),
                'gravity_projection':float(gn),'swivel_from_down_deg':float(np.rad2deg(np.arccos(np.clip(score/bend,-1,1))))}

    def posture(self,q):
        self.checker.fk(q);points=self.checker.d.xpos[self.bodies].copy()
        reports=[self.metrics(p)for p in points]
        return {'passed':all(r['passed']for r in reports),'arms':reports}

    def anchor_targets(self,seed_q,target_wrist,arms):
        self.checker.fk(seed_q);points=self.checker.d.xpos[self.bodies].copy();targets=points[:,1].copy()
        # Move elbow along with task geometry while retaining its seeded bend and axial fraction.
        for arm in arms:
            s,e,w=points[arm];v=w-s;f=float(np.dot(e-s,v)/np.dot(v,v));ep=e-s-f*v
            nv=target_wrist[arm]-s;length=np.linalg.norm(nv)
            if length<=self.axis_guard:continue
            u=nv/length;down=np.array([0.,0.,-1.]);down-=np.dot(down,u)*u;norm=np.linalg.norm(down)
            if norm>self.gravity_projection_guard:
                targets[arm]=s+f*nv+np.linalg.norm(ep)*down/norm
        return targets

    def interval_pass(self,points,radii):
        """Conservative sign of N=(e.g)(v.v)-(e.v)(g.v); g=(0,0,-1)."""
        for arm in range(2):
            s=interval(points[arm,0]-radii[arm,0],points[arm,0]+radii[arm,0])
            e=sub(interval(points[arm,1]-radii[arm,1],points[arm,1]+radii[arm,1]),s)
            v=sub(interval(points[arm,2]-radii[arm,2],points[arm,2]+radii[arm,2]),s)
            g=interval([0.,0.,-1.]);vv=dot(v,v);cg2=total(square((v[0][:2],v[1][:2])))
            ev=dot(e,v);num=sub(mul(dot(e,g),vv),mul(ev,dot(g,v)))
            bend_cross=dot(cross(e,v),cross(e,v))
            if vv[0]<=self.axis_guard**2:return False
            if cg2[0]<=self.gravity_projection_guard**2*vv[1]:return False
            if bend_cross[0]<=self.bend_guard**2*vv[1]:return False
            if num[0]<=self.score_guard*np.sqrt(max(0,float(vv[1]*cg2[1]))):return False
        return True

    def path(self,q0,q1,max_depth=12,max_nodes=4095):
        for q in (q0,q1):
            state=self.posture(q)
            if not state['passed']:return {'status':'rejected_elbow_branch','endpoint':state,'nodes':0}
        stack=[(np.array(q0),np.array(q1),0)];nodes=0;depth_used=0
        while stack:
            a,b,depth=stack.pop();nodes+=1;depth_used=max(depth_used,depth)
            if nodes>max_nodes:return {'status':'unknown_elbow_budget','nodes':nodes}
            state=self.posture((a+b)/2)
            if not state['passed']:return {'status':'rejected_elbow_branch','nodes':nodes,'posture':state}
            points=self.checker.d.xpos[self.bodies].copy();radii=self.levers@(np.abs(b-a)/2)
            if self.interval_pass(points,radii):continue
            if depth>=max_depth:return {'status':'unknown_elbow_resolution','nodes':nodes,'max_depth':depth_used}
            mid=(a+b)/2;stack.extend([(a,mid,depth+1),(mid,b,depth+1)])
        return {'status':'certified_elbow_down_path','nodes':nodes,'max_depth':depth_used}
