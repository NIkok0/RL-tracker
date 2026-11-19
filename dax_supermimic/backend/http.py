"""Same-origin HTTP routes and lifecycle for the existing8770 workspace."""
import argparse
import json
import signal
from pathlib import Path
from http.server import ThreadingHTTPServer
from ..common.paths import ROOT
from ..robot import JOINT_NAMES
from .reference_preview import handler
from .workspace import Workbench,EDIT_BODIES

WORKBENCH_WEB=ROOT/'dax_supermimic/web'

def workspace_handler(workspace):
    Base=handler(workspace)
    class Handler(Base):
        def do_GET(self):
            from urllib.parse import urlsplit,parse_qs
            parsed=urlsplit(self.path);query=parse_qs(parsed.query)
            try:
                if parsed.path=='/':self.reply((WORKBENCH_WEB/'workspace.html').read_bytes(),'text/html; charset=utf-8')
                elif parsed.path=='/data-generation':self.reply((WORKBENCH_WEB/'generation.html').read_bytes(),'text/html; charset=utf-8')
                elif parsed.path=='/data-generation/library':
                    html=(ROOT/'output/dax_synthetic_5h_20261103/viewer.html').read_text().replace('/api/catalog','/api/generation/catalog').replace('/api/frame?','/api/generation/frame?').replace('/quick_preview.mp4','/generation/quick_preview.mp4').replace('/physics_metrics.json','/generation/physics_metrics.json')
                    job=query.get('job',[None])[0]
                    if job:
                        directory=workspace.generation_pipeline._dir(job);release=json.loads((directory/'release/manifest.json').read_text())
                        html=html.replace('/api/generation/catalog','/api/generation/catalog?job='+job).replace('/api/generation/frame?','/api/generation/frame?job='+job+'&')
                        html=html.replace('18类 × 50段，共900段 / 5小时',str(len(release['clips']))+'段扰动变速合格参考 · 独立发布')
                        html=html.replace('几何筛查：每帧及帧间中点检查网格接触，拒绝超过1mm穿透的候选；保留模型碰撞排除项，不代表连续碰撞安全或真机可执行。','本批参考及记录状态经肘下/自碰撞硬检查；碰撞凸几何与结构排除表已绑定。仅名义仿真，跟踪精度单列诊断。')
                    self.reply(html.encode(),'text/html; charset=utf-8')
                elif parsed.path=='/api/generation/pipeline':self.reply(workspace.generation_pipeline.describe())
                elif parsed.path=='/api/generation/jobs':self.reply({'jobs':workspace.generation_pipeline.list_jobs()})
                elif parsed.path=='/api/generation/job':self.reply(workspace.generation_pipeline.status(query.get('id',[''])[0]))
                elif parsed.path=='/api/generation/catalog':self.reply(workspace.generated(job=query.get('job',[None])[0]))
                elif parsed.path=='/api/generation/frame':self.reply(workspace.generated(int(query.get('clip',['0'])[0]),int(query.get('frame',['0'])[0]),job=query.get('job',[None])[0]))
                elif parsed.path in ('/generation/quick_preview.mp4','/generation/physics_metrics.json'):
                    file=ROOT/'output/dax_synthetic_5h_20261103'/parsed.path.rsplit('/',1)[-1]
                    self.reply(file.read_bytes(),'video/mp4' if file.suffix=='.mp4' else 'application/json')
                elif parsed.path=='/api/workspace':self.reply({'joint_names':JOINT_NAMES,'joint_limits':workspace.limits.tolist(),'tracked_bodies':EDIT_BODIES,'clips':list(workspace.entries.values()),'checkpoints':workspace.checkpoints(),'default_checkpoint':next(iter(workspace.checkpoints()),None),'scope':'local simulation workspace; no hardware connection'})
                elif parsed.path=='/api/feedback':self.reply(workspace.feedback(query['clip'][0],int(query.get('frame',['0'])[0])))
                else:super().do_GET()
            except (ValueError,KeyError,OSError) as exc:self.reply({'error':str(exc)},status=400)

        def do_POST(self):
            # Local interactive writes are same-origin only; no arbitrary remote proxy.
            origin=self.headers.get('Origin')
            if origin and origin!=f'http://{self.headers.get("Host")}':self.reply({'error':'Cross-origin request rejected'},status=403);return
            if self.headers.get('Content-Type','').split(';')[0]!='application/json':self.reply({'error':'JSON required'},status=415);return
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=2_000_000:raise ValueError('Request size exceeds limit')
                payload=json.loads(self.rfile.read(size))
                if not isinstance(payload,dict):raise ValueError('JSON object required')
                actions={'/api/geometry':lambda:workspace.geometry(payload['q']),'/api/solve':lambda:workspace.solve(payload),
                         '/api/trajectory':lambda:workspace.trajectory(payload),'/api/simulation':lambda:workspace.simulation(payload),
                         '/api/generation/preview':lambda:workspace.generation_pipeline.preview(payload),
                         '/api/generation/create':lambda:workspace.generation_pipeline.create(payload),
                         '/api/generation/start':lambda:workspace.generation_start(payload)}
                if self.path not in actions:self.reply({'error':'Not found'},status=404);return
                self.reply(actions[self.path]())
            except (ValueError,KeyError,TypeError,OSError,EOFError) as exc:self.reply({'error':str(exc)},status=400)
    return Handler


def main(default_port=8770):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--port',type=int,default=default_port)
    parser.add_argument('--data',type=Path,default=ROOT/'data/dax_combined_front_training_v2_tailhold_20261105');args=parser.parse_args()
    workspace=Workbench(args.data);server=ThreadingHTTPServer(('127.0.0.1',args.port),workspace_handler(workspace))
    print(json.dumps({'url' :f'http://127.0.0.1:{args.port}','mode':'unified_workspace','hardware':False}),flush=True)
    def stop_service(*_):raise KeyboardInterrupt
    previous_term=signal.signal(signal.SIGTERM,stop_service)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:
        signal.signal(signal.SIGTERM,previous_term)
        server.server_close();workspace.simulation({'operation':'close'})


if __name__=="__main__":main()
