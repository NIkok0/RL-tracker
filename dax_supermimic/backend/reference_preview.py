"""Read-only browser preview of the upstream-format DAX motion reference."""
from __future__ import annotations
from functools import lru_cache
from http.server import BaseHTTPRequestHandler
import json,mimetypes
from pathlib import Path
from threading import Lock
import time
from urllib.parse import urlsplit,parse_qs
import mujoco
import numpy as np
from ..robot import ROOT,MJCF_PATH,JOINT_NAMES,TRACKED_BODIES,ROBOT_CONFIG,USD_PATH
from ..preflight import check
WEB=ROOT/'dax_supermimic/web'
VENDOR=WEB/'vendor'

class ReferencePreview:
    def __init__(self, directory, *, mjcf_path=None):
        self.directory=Path(directory).resolve()
        self._readiness_lock=Lock()
        self._readiness_result=None
        self._readiness_signature=None
        self._readiness_checked_at=0.0
        self.manifest=json.loads((self.directory/'manifest.json').read_text())
        self.entries={row['clip_id']:row for row in self.manifest['clips'] if row['split'] in ('train','val')}
        self.model=mujoco.MjModel.from_xml_path(str(mjcf_path or MJCF_PATH))
        self.qids=[self.model.jnt_qposadr[self.model.joint(name).id] for name in JOINT_NAMES]
        self.geoms=[i for i in range(self.model.ngeom) if self.model.geom_group[i]==2]
        meshes=[];pieces=[];offset=0
        for mid in sorted({int(self.model.geom_dataid[i]) for i in self.geoms}):
            if mid<0:continue
            first,count=int(self.model.mesh_vertadr[mid]),int(self.model.mesh_vertnum[mid])
            face_first,face_count=int(self.model.mesh_faceadr[mid]),int(self.model.mesh_facenum[mid])
            vertices=np.asarray(self.model.mesh_vert[first:first+count],dtype='<f4')
            faces=np.asarray(self.model.mesh_face[face_first:face_first+face_count],dtype='<u4')
            meshes.append({'id':mid,'vertex_offset':offset,'vertex_count':count,'face_offset':offset+vertices.nbytes,'face_count':face_count})
            pieces.extend((vertices.tobytes(),faces.tobytes()));offset+=vertices.nbytes+faces.nbytes
        self.binary=b''.join(pieces)
        self.scene={'meshes':meshes,'geoms':[{'id':i,'mesh':int(self.model.geom_dataid[i]),'rgba':self.model.geom_rgba[i].tolist()} for i in self.geoms],
                    'joint_names':JOINT_NAMES,'tracked_bodies':TRACKED_BODIES,'scope':'reference geometry only; mj_forward, no physics stepping or policy'}

    def readiness(self):
        """Reuse actual preflight evidence for 60 seconds; invalidate robot-file changes."""
        def fingerprint(path):
            try:
                value=Path(path).stat()
                return value.st_mtime_ns,value.st_size
            except FileNotFoundError:
                return None
        signature=tuple(fingerprint(path) for path in (ROBOT_CONFIG,USD_PATH))
        with self._readiness_lock:
            now=time.monotonic()
            if (self._readiness_result is None or signature!=self._readiness_signature
                    or now-self._readiness_checked_at>=60.0):
                result=check()
                result['cache']={'max_age_seconds':60,'checked_at_unix_s':time.time()}
                self._readiness_result=result
                self._readiness_signature=signature
                self._readiness_checked_at=time.monotonic()
            return self._readiness_result

    @lru_cache(maxsize=3)
    def clip(self, name):
        row=self.entries[name]
        return np.loadtxt(self.directory/row['file'],delimiter=',',dtype=np.float32)

    def state(self,name,frame):
        values=self.clip(name);frame=max(0,min(int(frame),len(values)-1))
        data=mujoco.MjData(self.model);data.qpos[self.qids]=values[frame,7:];mujoco.mj_forward(self.model,data)
        return {'clip':name,'frame':frame,'frames':len(values),'time_s':frame*.02,
                'q':values[frame,7:].tolist(),'geom_positions':data.geom_xpos[self.geoms].tolist(),
                'geom_rotations':data.geom_xmat[self.geoms].reshape(-1,3,3).tolist(),
                'scope':'requested joint reference geometry; no policy or physical validation'}

def handler(preview):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def reply(self,value,mime='application/json; charset=utf-8',status=200):
            if not isinstance(value,bytes):value=json.dumps(value,ensure_ascii=False,allow_nan=False).encode()
            self.send_response(status);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(value)));self.send_header('Cache-Control','no-store');self.end_headers()
            try:self.wfile.write(value)
            except (BrokenPipeError,ConnectionResetError):pass
        def do_GET(self):
            parsed=urlsplit(self.path);path=parsed.path
            try:
                if path=='/api/reference':
                    self.reply({'readiness':preview.readiness(),'clips':list(preview.entries.values()),'reference_signal':'requested_q','future_goal_frames':10,'goal_size':510,'physics_running':False})
                elif path=='/api/scene':self.reply(preview.scene)
                elif path=='/api/meshes.bin':self.reply(preview.binary,'application/octet-stream')
                elif path=='/api/frame':
                    query=parse_qs(parsed.query);self.reply(preview.state(query['clip'][0],int(query.get('frame',['0'])[0])))
                else:
                    if path.startswith('/vendor/'):base=VENDOR;relative=path.removeprefix('/vendor/')
                    else:base=WEB;relative='index.html' if path=='/' else path.lstrip('/')
                    file=(base/relative).resolve()
                    if not file.is_relative_to(base.resolve()) or not file.is_file():self.reply({'error':'not found'},status=404);return
                    self.reply(file.read_bytes(),mimetypes.guess_type(file.name)[0] or 'application/octet-stream')
            except (ValueError,KeyError) as exc:self.reply({'error':str(exc)},status=400)
    return Handler
