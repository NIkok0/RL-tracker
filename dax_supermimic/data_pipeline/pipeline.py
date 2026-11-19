"""Persistent data-generation workflow; planning is available, execution adapters are explicit.

No training/hardware transport. UI can create verified plans, never submit validation receipts.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import re
from threading import RLock
import uuid

from ..common.paths import ROOT
from ..common.artifacts import atomic,sha
from .records import physical_trace_binding

STAGES=('prepare','geometry','validate','publish')
LABELS={'prepare':'来源、目标与初态配方','geometry':'轨迹生成与几何约束','validate':'速度时序与物理筛查','publish':'独立发布与可视化'}
REQUIRED_GATES=('source_lineage','joint_limits','elbow_down_seed','elbow_down_path',
                'self_collision_path','timing_limits','resampled_geometry','actual_state_geometry','physics_tracking')



def implementation_hashes():
    paths=[]
    for package in ('data_pipeline','robot_sim','common'):
        paths.extend(sorted((ROOT/'dax_supermimic'/package).glob('*.py')))
    return {str(p.relative_to(ROOT)):sha(p)for p in paths}


class Pipeline:
    def __init__(self,root=ROOT,jobs=None):
        self.root=Path(root).resolve();self.jobs=Path(jobs or self.root/'output/data_generation_jobs').resolve()
        self.lock=RLock()
        self.source=self.root/'data/dax_synthetic_5h_v1_20261103'
        self.asset=self.root/'data/dax_mixed_teleop_synthetic_v3_20261103/asset.json'

    def configuration(self,values=None):
        values=values or {}
        if not isinstance(values,dict):raise ValueError('配置必须为JSON对象')
        allowed={'method','source_ids','seed','variants_per_clip','speed_factors','joint_perturbation_rad','perturbation_levels_rad','screening_profile','limits','geometry_screening','candidate_count'}
        if set(values)-allowed:raise ValueError('未知配置字段')
        cfg={'method':'augment_existing','source_ids':[],'seed':20261104,'variants_per_clip':1,
             'speed_factors':[.5,1.,1.5],'joint_perturbation_rad':.03,'perturbation_levels_rad':None,'screening_profile':'full_quality','candidate_count':None,
             'limits':{'velocity_rad_s':None,'acceleration_rad_s2':None,'jerk_rad_s3':None,
                       'tracking_rmse_m':None,'tracking_p95_m':None},**values}
        if cfg['method']!='augment_existing':raise ValueError('当前只登记已有数据增强方式')
        for key,lo,hi in [('seed',0,2**31-1),('variants_per_clip',1,20)]:
            if type(cfg[key])is not int or not lo<=cfg[key]<=hi:raise ValueError('无效'+key)
        if cfg['candidate_count']is not None and(type(cfg['candidate_count'])is not int or not 1<=cfg['candidate_count']<=100000):raise ValueError('候选总数应为1至100000或留空')
        speed=cfg['speed_factors']
        if not isinstance(speed,list)or not 1<=len(speed)<=10:raise ValueError('速度档应为1至10个数值')
        if any(type(v)not in(int,float)or not math.isfinite(v)or not .1<=v<=3 for v in speed):raise ValueError('速度档范围0.1至3')
        if len(set(speed))!=len(speed):raise ValueError('速度档重复')
        amp=cfg['joint_perturbation_rad']
        if type(amp)not in(int,float)or not math.isfinite(amp)or not 0<=amp<=.3:raise ValueError('扰动幅度应在0至0.3rad')
        levels=cfg['perturbation_levels_rad']
        if levels is not None:
            if not isinstance(levels,list)or not 1<=len(levels)<=5 or len(set(levels))!=len(levels):raise ValueError('扰动档必须唯一，1至5档')
            if any(type(v)not in(int,float)or not math.isfinite(v)or not 0<v<=.3 for v in levels):raise ValueError('扰动档范围0至0.3rad')
        if cfg['screening_profile']not in('full_quality','reference_plus_physics_diagnostics'):raise ValueError('未知筛查口径')
        ids=cfg['source_ids']
        if not isinstance(ids,list)or any(not isinstance(x,str)for x in ids)or len(ids)!=len(set(ids)):raise ValueError('来源片段ID必须唯一')
        if not isinstance(cfg['limits'],dict)or set(cfg['limits'])!=set(('velocity_rad_s','acceleration_rad_s2','jerk_rad_s3','tracking_rmse_m','tracking_p95_m')):raise ValueError('约束字段不完整')
        for v in cfg['limits'].values():
            if v is not None and(type(v)not in(int,float)or not math.isfinite(v)or v<=0):raise ValueError('约束值必须为正数或待定')
        # Mandatory gates cannot be disabled or replaced by client config.
        from .quality import screening_config
        cfg['geometry_screening']=screening_config(values.get('geometry_screening'))
        cfg['hard_constraints']={'elbow_down':True,'self_collision':True,'both_arms':True,
              'post_resampling_recheck':True,'physical_state_recheck':True}
        return cfg

    def _inputs(self,cfg,verify=True):
        manifest_path=self.source/'manifest.json';manifest=json.loads(manifest_path.read_text())
        rows=manifest['clips'];byid={r['clip_id']:r for r in rows}
        ids=cfg['source_ids']or list(byid)
        if any(i not in byid for i in ids):raise ValueError('来源片段不存在')
        selected=[byid[i]for i in ids]
        for r in selected:
            p=(self.source/r['file']).resolve()
            if not p.is_relative_to(self.source.resolve()):raise ValueError('来源路径越界')
            if verify and sha(p)!=r['sha256']:raise ValueError('来源文件哈希变化：'+r['clip_id'])
        asset=json.loads(self.asset.read_text())
        if verify:
            for path,expected in asset['hashes'].items():
                p=(self.root/path).resolve()
                if not p.is_relative_to(self.root):raise ValueError('模型路径越界')
                if sha(p)!=expected:raise ValueError('模型文件哈希变化')
        return selected,{'source_manifest':str(manifest_path.relative_to(self.root)),
                'source_manifest_sha256':sha(manifest_path),'asset':str(self.asset.relative_to(self.root)),
                'asset_sha256':sha(self.asset),'joint_names':manifest['joint_names'],
                'fixed_lower_rad':asset['fixed_rad'],'source_files':[{k:r[k]for k in ('clip_id','file','sha256','frames')}for r in selected]}

    def describe(self):
        return {'schema':'dax_generation_workflow_v1','stages':[{'id':s,'label':LABELS[s],
             'adapter':'available',
             'depends_on':None if i==0 else STAGES[i-1]}for i,s in enumerate(STAGES)],
             'methods':[{'id':'augment_existing','label':'已有900段：轨迹扰动＋速度变化',
                 'status':'local_adapter_available','speed_semantics':'实际执行器支持0.1至3倍；保留末端，变速后重新检查几何、时序与物理状态',
                 'perturbation_semantics':'平滑关节扰动＋锚点/肘下投影，实际算法在轨迹adapter实现；不是独立逐帧白噪声'}],
             'geometry_screening_note':'全部控制帧＋按关节位移补点；显式碰撞对和逐对净空；离散近似，不承诺连续无碰撞',
             'required_gates':list(REQUIRED_GATES),'default_config':self.configuration(),
             'execution_enabled':True,'local_runner_available':True,
             'execution_note':'扰动、变速与物理筛查已接通本地作业；页面按输入建立计划，点击开始后由后台执行器运行；打开页面不会自动生成。'}

    def preview(self,values=None):
        cfg=self.configuration(values);rows,binding=self._inputs(cfg)
        source_duration=sum((r['frames']-1)*.02 for r in rows)
        chosen=Pipeline._prepare(self,None,{'config':cfg,'bindings':binding})['recipes']
        frames={r['clip_id']:r['frames']for r in rows}
        return {'config':cfg,'bindings':binding,'source_clips':len(rows),
             'workflow_sha256':sha(__file__),
             'implementation_sha256':implementation_hashes(),
             'planned_candidates':len(chosen),
             'estimated_reference_seconds_before_rejection':sum((frames[r['parent_clip']]-1)*.02/r['speed_factor']for r in chosen),
             'estimate_note':'仅按速度倍率的计划时长，不是已生成/已通过数据量',
             'unresolved_limits':[k for k,v in cfg['limits'].items()if v is None and(cfg['screening_profile']=='full_quality'or k in('velocity_rad_s','acceleration_rad_s2'))],
             'diagnostic_limits':[k for k,v in cfg['limits'].items()if v is None and cfg['screening_profile']=='reference_plus_physics_diagnostics'and k not in('velocity_rad_s','acceleration_rad_s2')],
             'source_verification':'passed','stage1_note':'本阶段只冻结候选配方；每个seed仍须在geometry阶段做肘下与碰撞合法性筛选'}

    def _dir(self,job_id):
        if not isinstance(job_id,str)or not re.fullmatch(r'[0-9a-f]{32}',job_id):raise ValueError('无效任务ID')
        path=self.jobs/job_id
        if not path.is_dir():raise ValueError('任务不存在')
        return path

    def create(self,values=None):
        with self.lock:
            plan=self.preview(values);job_id=uuid.uuid4().hex;folder=self.jobs/job_id;folder.mkdir(parents=True)
            atomic(folder/'plan.json',plan)
            states={s:{'status':'pending','artifact':None}for s in STAGES}
            atomic(folder/'state.json',{'schema':'dax_generation_job_v1','job_id':job_id,'status':'planned',
                 'plan_sha256':sha(folder/'plan.json'),
                 'stages':states,'events':[{'stage':None,'event':'plan_frozen'}]})
            self.execute(job_id,'prepare')
            return self.status(job_id)

    def status(self,job_id):
        folder=self._dir(job_id);state=json.loads((folder/'state.json').read_text());plan=json.loads((folder/'plan.json').read_text())
        return {**state,'source_clips':plan['source_clips'],'planned_candidates':plan['planned_candidates'],
             'unresolved_limits':plan['unresolved_limits'],'plan_path':str(folder/'plan.json'),
             'config':plan['config'],
             'progress':json.loads((folder/'progress.json').read_text())if (folder/'progress.json').is_file()else None,
             'execution':json.loads((folder/'execution.json').read_text())if (folder/'execution.json').is_file()else None,
             'release_path':str(folder/'release')if (folder/'release/manifest.json').is_file()else None}

    def list_jobs(self):
        if not self.jobs.exists():return []
        return [self.status(p.name)for p in sorted(self.jobs.iterdir(),key=lambda p:p.stat().st_mtime,reverse=True)
                if p.is_dir()and re.fullmatch(r'[0-9a-f]{32}',p.name)and(p/'state.json').is_file()][:30]

    def _check_bindings(self,plan):
        b=plan['bindings']
        if plan['workflow_sha256']!=sha(__file__):raise ValueError('流程实现已变化，需创建新计划')
        for name,expected in plan['implementation_sha256'].items():
            if sha(ROOT/name)!=expected:raise ValueError('执行器实现已变化，需创建新计划')
        if sha(self.root/b['source_manifest'])!=b['source_manifest_sha256']or sha(self.root/b['asset'])!=b['asset_sha256']:
            raise ValueError('来源或模型已变化，需创建新计划')
        _,now=self._inputs(plan['config'])
        if now!=b:raise ValueError('任务绑定来源变化')

    def execute(self,job_id,stage,adapter=None):
        """Local adapter interface. HTTP never supplies adapters or arbitrary artifact paths."""
        if stage not in STAGES:raise ValueError('未知阶段')
        with self.lock:
            folder=self._dir(job_id);plan=json.loads((folder/'plan.json').read_text());state=json.loads((folder/'state.json').read_text())
            if sha(folder/'plan.json')!=state['plan_sha256']:raise ValueError('冻结计划被修改')
            if state['stages'][stage]['status']=='completed':return self.status(job_id)
            i=STAGES.index(stage)
            if i and state['stages'][STAGES[i-1]]['status']!='completed':raise ValueError('前置阶段未完成')
            self._check_bindings(plan)
            # Completed upstream artifacts are immutable and verified before downstream work.
            for earlier in STAGES[:i]:
                rec=state['stages'][earlier];p=folder/rec['artifact']
                if sha(p)!=rec['sha256']:raise ValueError('前置产物变化')
            if stage in ('geometry','validate')and adapter is None:
                state['stages'][stage]['status']='blocked';state['stages'][stage]['reason']='adapter_not_connected'
                state['status']='blocked';state['events'].append({'stage':stage,'event':'blocked','reason':'adapter_not_connected'})
                atomic(folder/'state.json',state);return self.status(job_id)
            unresolved=plan['unresolved_limits']
            if plan['config']['screening_profile']=='reference_plus_physics_diagnostics':
                unresolved=[k for k in unresolved if k in('velocity_rad_s','acceleration_rad_s2')]
            if stage=='validate'and unresolved:raise ValueError('时序/物理筛查阈值尚未冻结')
            state['stages'][stage]['status']='running';state['status']='running';atomic(folder/'state.json',state)
            try:
                if stage=='prepare':result=self._prepare(folder,plan)
                elif stage=='publish':result=self._publish(folder,plan,state)
                else:
                    previous=json.loads((folder/state['stages'][STAGES[i-1]]['artifact']).read_text())
                    result=adapter(plan,previous,folder)
                    if not isinstance(result,dict):raise ValueError('执行器须返回阶段产物对象')
                    self._check_output(stage,result,folder,plan,previous)
                artifact=folder/(stage+'.json');atomic(artifact,result)
                state['stages'][stage]={'status':'completed','artifact':artifact.name,'sha256':sha(artifact)}
                state['status']='published'if stage=='publish'else'awaiting_adapter'if stage=='prepare'else'ready'
                state['events'].append({'stage':stage,'event':'completed'})
            except Exception as e:
                state['stages'][stage].update(status='failed',reason=str(e));state['status']='failed'
                state['events'].append({'stage':stage,'event':'failed','reason':str(e)})
                atomic(folder/'state.json',state);raise
            atomic(folder/'state.json',state);return self.status(job_id)

    def _prepare(self,folder,plan):
        cfg=plan['config'];recipes=[]
        for row in plan['bindings']['source_files']:
            for variant in range(cfg['variants_per_clip']):
                for amplitude in cfg['perturbation_levels_rad']or[cfg['joint_perturbation_rad']]:
                    group=f'{row["clip_id"]}:{variant}:{amplitude}:{cfg["seed"]}'
                    for speed in cfg['speed_factors']:
                        key=group+':'+str(speed)
                        recipes.append({'candidate_id':hashlib.sha256(key.encode()).hexdigest()[:24],
                           'geometry_group':hashlib.sha256(group.encode()).hexdigest()[:24],
                           'parent_clip':row['clip_id'],'parent_file':row['file'],'parent_sha256':row['sha256'],'variant':variant,
                           'seed':int.from_bytes(hashlib.sha256(group.encode()).digest()[:4],'little'),
                           'speed_factor':speed,'joint_perturbation_rad':amplitude,
                           'initial_state_status':'must_pass_elbow_and_collision_gate','status':'recipe_only'})
        if cfg.get('candidate_count')is not None and cfg['candidate_count']<len(recipes):
            import random
            rng=random.Random(cfg['seed']);groups={}
            for row in recipes:groups.setdefault(row['parent_clip'],[]).append(row)
            sources=list(groups);rng.shuffle(sources)
            for rows in groups.values():rng.shuffle(rows)
            selected=[]
            while len(selected)<cfg['candidate_count']:
                for source in sources:
                    if groups[source]:selected.append(groups[source].pop())
                    if len(selected)==cfg['candidate_count']:break
            recipes=selected
        return {'kind':'candidate_recipes','recipes':recipes,'generated_trajectories':0,
                'legal_initial_states_verified':False,'note':'候选配方完成；未生成轨迹、未宣称起点合法'}

    def _check_output(self,stage,result,folder,plan,previous):
        records=result.get('candidates')
        if not isinstance(records,list):raise ValueError('缺少逐候选记录')
        upstream=previous['recipes']if stage=='geometry'else previous['candidates']
        expected={r['candidate_id']:r for r in upstream};ids=[r.get('candidate_id')for r in records]
        if len(ids)!=len(set(ids))or set(ids)!=set(expected):raise ValueError('不得丢弃或重复前置候选；失败也必须记录')
        for r in records:
            if r.get('status')not in('accepted','rejected','unknown'):raise ValueError('未知候选状态')
            parent=expected[r['candidate_id']]
            if r.get('parent_sha256')!=parent['parent_sha256']:raise ValueError('来源链不匹配')
            if stage=='validate'and parent['status']!='accepted'and r['status']=='accepted':raise ValueError('不能放行几何未通过的候选')
            if r['status']!='accepted':
                if not r.get('reason'):raise ValueError('拒绝/未知须附原因')
                continue
            required=REQUIRED_GATES[:5]if stage=='geometry'else REQUIRED_GATES
            trajectory=(folder/r.get('trajectory'if stage=='geometry'else'csv','')).resolve()
            if not trajectory.is_relative_to(folder.resolve())or not trajectory.is_file():raise ValueError('当前轨迹产物缺失/越界')
            trajectory_hash=sha(trajectory)
            pair_table=folder/'collision_pairs.json'
            if not pair_table.is_file():raise ValueError('缺少显式碰撞对表')
            pair_hash=sha(pair_table)
            trace_binding=None
            if stage=='validate':
                trace_binding=physical_trace_binding(folder,r.get('physics',{}),trajectory)
                if r['physics'].get('trace_binding')!=trace_binding:raise ValueError('物理轨迹完整性绑定不匹配')
            evidence=r.get('evidence',{})
            for gate in required:
                receipt=evidence.get(gate,{})
                if receipt.get('passed')is not True:raise ValueError('必经约束未通过：'+gate)
                p=(folder/receipt.get('file','')).resolve()
                if not p.is_relative_to(folder.resolve())or not p.is_file()or sha(p)!=receipt.get('sha256'):raise ValueError('证据文件缺失/变化')
                content=json.loads(p.read_text())
                expected_receipt={'gate':gate,'candidate_id':r['candidate_id'],'parent_sha256':r['parent_sha256'],
                      'trajectory_sha256':trajectory_hash,'asset_sha256':plan['bindings']['asset_sha256'],
                      'plan_sha256':sha(folder/'plan.json'),'collision_pairs_sha256':pair_hash,'passed':True}
                if any(content.get(k)!=v for k,v in expected_receipt.items()):raise ValueError('约束证据与当前候选/轨迹/配置不匹配')
                if stage=='validate'and gate in('actual_state_geometry','physics_tracking')and content.get('physical_trace')!=trace_binding:
                    raise ValueError('物理证据与当前回放轨迹不匹配')
            if stage=='validate':
                import numpy as np
                p=(folder/r.get('csv','')).resolve();side=(folder/r.get('full20','')).resolve()
                if any(not x.is_relative_to(folder.resolve())or not x.is_file()for x in (p,side)):raise ValueError('数据产物缺失/越界')
                q=np.loadtxt(p,delimiter=',',ndmin=2)
                if q.shape[1]!=25 or len(q)<2 or not np.isfinite(q).all():raise ValueError('无效参考CSV')
                if not np.all(q[:,:6]==0)or not np.all(q[:,6]==1):raise ValueError('固定根CSV约定不符')
                with np.load(side,allow_pickle=False)as z:
                    if z['joint_names'].tolist()!=['waist_joint0','waist_joint1']+plan['bindings']['joint_names']:raise ValueError('具名顺序不符')
                    if z['reference_q'].shape!=(len(q),20)or not np.array_equal(z['reference_q'][:,2:].astype('f4'),q[:,7:].astype('f4')):raise ValueError('旁车与CSV不符')
                    if not np.allclose(z['reference_q'][:,:2],plan['bindings']['fixed_lower_rad'],rtol=0,atol=1e-12):raise ValueError('固定下腰值不符')
                    if not np.allclose(z['time_s'],np.arange(len(q))*.02,rtol=0,atol=1e-9):raise ValueError('必须50Hz')
                r['csv_sha256']=sha(p);r['full20_sha256']=sha(side)

    def _publish(self,folder,plan,state):
        previous=json.loads((folder/state['stages']['geometry']['artifact']).read_text())
        result=json.loads((folder/state['stages']['validate']['artifact']).read_text())
        self._check_output('validate',result,folder,plan,previous)
        accepted=[r for r in result['candidates']if r['status']=='accepted']
        if not accepted:raise ValueError('无全部准入通过的数据，不创建发布')
        release=folder/'release'
        if release.exists():raise ValueError('发布目录已存在，拒绝覆盖')
        import shutil
        staging=folder/'release.building';staging.mkdir()
        shutil.copyfile(folder/'plan.json',staging/'plan.json')
        shutil.copyfile(folder/'collision_pairs.json',staging/'collision_pairs.json')
        if sha(staging/'collision_pairs.json')!=sha(folder/'collision_pairs.json'):raise ValueError('碰撞对表复制哈希不符')
        (staging/'evidence').mkdir()
        (staging/'physical_traces').mkdir()
        for r in accepted:
            key=r['candidate_id']
            for field,suffix in [('csv','.csv'),('full20','.npz')]:
                source=(folder/r[field]).resolve();dest=staging/(key+suffix);shutil.copyfile(source,dest)
                if sha(dest)!=r[field+'_sha256']:raise ValueError('发布复制哈希不符')
            trace_name='physical_traces/'+key+'.npz'
            shutil.copyfile(folder/r['physics']['physical_npz'],staging/trace_name)
            if sha(staging/trace_name)!=r['physics']['trace_binding']['physical_npz_sha256']:raise ValueError('物理轨迹复制哈希不符')
            r['physics']['physical_npz']=trace_name
            for gate,receipt in r['evidence'].items():
                name='evidence/'+key+'_'+gate+'.json';shutil.copyfile(folder/receipt['file'],staging/name)
                if sha(staging/name)!=receipt['sha256']:raise ValueError('约束证据复制哈希不符')
                receipt['file']=name
        manifest={'schema':'dax_generated_release_v2','job_id':folder.name,'fps':50,'training_included':False,
                 'screening_profile':plan['config']['screening_profile'],
                 'collision_pairs':{'file':'collision_pairs.json','sha256':sha(folder/'collision_pairs.json')},
                 'geometry_screening':plan['config']['geometry_screening'],
                 'joint_names':plan['bindings']['joint_names'],'source_bindings':plan['bindings'],
                 'clips':accepted,'rejected_or_unknown':[r for r in result['candidates']if r['status']!='accepted']}
        for r in manifest['clips']:
            r['csv']=r['candidate_id']+'.csv';r['full20']=r['candidate_id']+'.npz'
        atomic(staging/'manifest.json',manifest);staging.rename(release)
        return {'kind':'independent_release','manifest':'release/manifest.json','sha256':sha(release/'manifest.json'),
                'accepted':len(accepted),'training_included':False,'publication_scope':'local job directory; no training dataset replacement'}
