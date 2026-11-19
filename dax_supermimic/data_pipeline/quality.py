"""Sampled motion screening with explicit pair clearances, not continuous certification."""
from __future__ import annotations
import math
from collections import Counter
import numpy as np

NUMERIC_GUARD_M=1e-6  # Numerical ambiguity, separate from application clearance.
DEFAULT_SCREENING={'method':'sampled_joint_step_v1','max_joint_step_rad':.01,
                   'max_samples':200000,'collision_margin_m':0.,'pair_margins_m':[]}


def screening_config(value=None):
    if value is not None and not isinstance(value,dict):raise ValueError('几何筛查配置必须为对象')
    value=value or {}
    if set(value)-set(DEFAULT_SCREENING):raise ValueError('未知几何筛查字段')
    cfg={**DEFAULT_SCREENING,**value,'pair_margins_m':value.get('pair_margins_m',[])}
    if cfg['method']!='sampled_joint_step_v1':raise ValueError('未知几何筛查方法')
    for key,low,high in [('max_joint_step_rad',1e-5,.1),('collision_margin_m',0.,.1)]:
        v=cfg[key]
        if type(v)not in(int,float)or not math.isfinite(v)or not low<=v<=high:raise ValueError('无效'+key)
    if type(cfg['max_samples'])is not int or not 2<=cfg['max_samples']<=2000000:raise ValueError('无效max_samples')
    if not isinstance(cfg['pair_margins_m'],list):raise ValueError('逐对裕量必须为列表')
    seen=set();overrides=[]
    for row in cfg['pair_margins_m']:
        if not isinstance(row,dict)or set(row)!={'geoms','margin_m'}:raise ValueError('无效逐对裕量')
        names=row['geoms'];margin=row['margin_m']
        if not isinstance(names,list)or len(names)!=2 or any(not isinstance(n,str)or not n for n in names)or names[0]==names[1]:raise ValueError('无效碰撞几何体对')
        key=tuple(sorted(names))
        if key in seen:raise ValueError('碰撞几何体对重复')
        if type(margin)not in(int,float)or not math.isfinite(margin)or not 0<=margin<=.1:raise ValueError('无效逐对裕量')
        seen.add(key);overrides.append({'geoms':list(key),'margin_m':margin})
    cfg['pair_margins_m']=sorted(overrides,key=lambda x:x['geoms'])
    return cfg


def pair_policy(checker,config):
    """Only inherited explicit exclusions/same-body suppression; never learn exclusions."""
    cfg=screening_config(config);pairs=checker.metadata['checked_pairs']
    keys={tuple(sorted(p))for p in pairs}
    overrides={tuple(r['geoms']):r['margin_m']for r in cfg['pair_margins_m']}
    if set(overrides)-keys:raise ValueError('逐对裕量引用未知或已排除的碰撞对')
    margins=np.array([overrides.get(tuple(sorted(p)),cfg['collision_margin_m'])for p in pairs])
    if not len(pairs):raise ValueError('碰撞检查对表不能为空')
    return margins,{'method':cfg['method'],'checked_pairs':[{'geoms':p,'clearance_m':float(m)}for p,m in zip(pairs,margins)],
         'explicit_body_exclusions':checker.metadata.get('explicit_body_exclusions',[]),
         'suppressed_pairs':checker.metadata.get('suppressed_pairs',[]),
         'exclusion_provenance':'inherited model explicit body exclusions and same-body only; no automatic exclusions',
         'numeric_guard_m':NUMERIC_GUARD_M,'geometry_scope':'MuJoCo compiled convex geometry',
         'decision':'distance > pair clearance + numerical guard at every checked sample',
         'scope':'discrete approximation; no continuous collision or solver-substep guarantee'}


def screen_polyline(q,checker,elbow,config=None,policy=None):
    cfg=screening_config(config)
    margins,table=policy if policy is not None else pair_policy(checker,cfg)
    q=np.asarray(q)
    base={'method':cfg['method'],'max_joint_step_rad':cfg['max_joint_step_rad'],
          'numeric_guard_m':NUMERIC_GUARD_M,'collision_margin_m':cfg['collision_margin_m'],
          'scope':table['scope']}
    def fail(reason,category,**detail):
        return {**base,'passed':False,'status':'unknown'if category in('inspection_incomplete','numerical_uncertainty')else'rejected',
                'reason':reason,'failure_category':category,**detail}
    if q.ndim!=2 or q.shape[1]!=len(checker.jids)or len(q)<2 or not np.isfinite(q).all():
        return fail('invalid_geometry_input','inspection_incomplete')
    low=np.array([checker.m.jnt_range[j,0]for j in checker.jids]);high=np.array([checker.m.jnt_range[j,1]for j in checker.jids])
    invalid=np.flatnonzero(np.any((q<low)|(q>high),axis=1))
    if len(invalid):return fail('joint_limits','confirmed_violation',source_frame=int(invalid[0]),checked_samples=0)
    steps=np.maximum(1,np.ceil(np.max(np.abs(np.diff(q,axis=0)),axis=1)/cfg['max_joint_step_rad']).astype(np.int64))
    needed=1+sum(int(n)for n in steps)
    if needed>cfg['max_samples']:
        return fail('sample_budget','inspection_incomplete',required_samples=needed,checked_samples=0)
    count=0;min_gap=math.inf;min_score=math.inf;minimum=None
    def samples():
        yield 0,0.,q[0]
        for frame,n in enumerate(steps):
            for k in range(1,int(n)+1):
                fraction=k/int(n)
                yield frame,float(fraction),q[frame+1]if k==n else q[frame]+fraction*(q[frame+1]-q[frame])
    for frame,fraction,state in samples():
        count+=1;where={'segment':frame,'fraction':fraction,'checked_samples':count,'required_samples':needed}
        if np.any(state<low)or np.any(state>high):return fail('joint_limits','confirmed_violation',**where)
        ds=checker.distances(state)
        # The reusable checker returns bounding-sphere lower bounds for positive gaps.
        # Refine bounds close to a configured clearance before filtering a sample.
        if hasattr(checker,'a'):
            import mujoco
            for i in np.flatnonzero((ds>0)&(ds<=margins+NUMERIC_GUARD_M)):
                ds[i]=mujoco.mj_geomDistance(checker.m,checker.d,int(checker.a[i]),int(checker.b[i]),10.,None)
        if ds.shape!=margins.shape or not np.isfinite(ds).all():return fail('collision_nonfinite','inspection_incomplete',**where)
        gap=ds-margins;idx=int(gap.argmin())
        if float(gap[idx])<min_gap:
            min_gap=float(gap[idx]);minimum={'geoms':table['checked_pairs'][idx]['geoms'],
                'distance_lower_bound_m':float(ds[idx]),'clearance_m':float(margins[idx]),'segment':frame,'fraction':fraction}
        if gap[idx]<=NUMERIC_GUARD_M:
            penetration=ds[idx]<-NUMERIC_GUARD_M
            return fail('collision_penetration'if penetration else'collision_clearance_filter',
                'confirmed_violation'if penetration else'clearance_filter',limiting_pair=minimum,
                geometric_interpretation='model_penetration'if penetration else'numerical_boundary_or_insufficient_clearance',**where)
        # distances() already updated FK; avoid another FK for elbow metrics.
        arms=[elbow.metrics(points)for points in checker.d.xpos[elbow.bodies]]
        if not all(r['passed']for r in arms):
            confirmed=any(r.get('score_m')is not None and r['score_m'] < -elbow.score_guard for r in arms)
            return fail('elbow_up'if confirmed else'elbow_boundary','confirmed_violation'if confirmed else'numerical_uncertainty',arms=arms,**where)
        min_score=min(min_score,min(r['score_m']for r in arms))
    return {**base,'passed':True,'status':'accepted','source_frames':len(q),'checked_samples':count,
            'inserted_samples':count-len(q),'required_samples':needed,'limiting_pair':minimum,
            'minimum_clearance_gap_lower_bound_m':min_gap,'min_sampled_elbow_down_score_m':min_score}


def pair_diagnostics(records,fields):
    """Count first/limiting pair by stage; evidence for review, never exclusion rules."""
    grouped={}
    for row in records:
        for field in fields:
            report=row.get(field,{})
            pair=report.get('limiting_pair')or{}
            if not pair.get('geoms'):continue
            key=(field,*sorted(pair['geoms']))
            item=grouped.setdefault(key,{'stage':field,'geoms':list(key[1:]),'candidates':0,'outcomes':{},'examples':[]})
            item['candidates']+=1
            outcome='accepted'if report['passed']else report.get('failure_category','unknown')
            item['outcomes'][outcome]=item['outcomes'].get(outcome,0)+1
            if len(item['examples'])<5:item['examples'].append({'candidate_id':row['candidate_id'],'finding':pair})
    return {'scope':'one first-failing or minimum-clearance pair per candidate/stage; not all contacts; no automatic exclusions',
            'pairs':sorted(grouped.values(),key=lambda x:(-x['candidates'],x['stage'],x['geoms']))}

def failed_check(reason,category,**details):
    return {'passed':False,'status':'rejected'if category=='confirmed_violation'else'unknown',
            'failure_category':category,'reason':reason,**details}


def report_status(report):
    return 'accepted'if report['passed']else report['status']


V2_ELBOW_ANGLE_GUARD_DEG = 0.01
V2_SCREENING_FIELDS = set(DEFAULT_SCREENING) | {'elbow_policy', 'extra_pair_exclusions'}


def screening_config_v2(value=None):
    """Validate the independent v2 policy without changing the v1 contract."""
    if value is not None and not isinstance(value, dict):
        raise ValueError('几何筛查配置必须为对象')
    value = value or {}
    if set(value) - V2_SCREENING_FIELDS:
        raise ValueError('未知v2几何筛查字段')
    if value.get('method', 'sampled_joint_step_v2') != 'sampled_joint_step_v2':
        raise ValueError('v2筛查方法必须为sampled_joint_step_v2')
    core = {key: value.get(key, default) for key, default in DEFAULT_SCREENING.items()}
    core['method'] = 'sampled_joint_step_v1'
    core = screening_config(core)
    elbow = value.get('elbow_policy', {'mode': 'max_swivel', 'max_swivel_from_down_deg': 150})
    if (not isinstance(elbow, dict) or set(elbow) != {'mode', 'max_swivel_from_down_deg'}
            or elbow.get('mode') != 'max_swivel'):
        raise ValueError('无效elbow_policy')
    threshold = elbow['max_swivel_from_down_deg']
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 < threshold <= 180:
        raise ValueError('无效max_swivel_from_down_deg')
    extra = value.get('extra_pair_exclusions', [])
    if not isinstance(extra, list):
        raise ValueError('extra_pair_exclusions必须为列表')
    exclusions = []
    seen = set()
    for row in extra:
        if not isinstance(row, dict) or set(row) != {'geoms', 'reason'}:
            raise ValueError('无效extra_pair_exclusions条目')
        names, reason = row['geoms'], row['reason']
        if (not isinstance(names, list) or len(names) != 2
                or any(not isinstance(name, str) or not name for name in names)
                or names[0] == names[1] or not isinstance(reason, str) or not reason.strip()):
            raise ValueError('无效几何排除对或理由')
        pair = tuple(sorted(names))
        if pair in seen:
            raise ValueError('extra_pair_exclusions包含重复几何对')
        seen.add(pair)
        exclusions.append({'geoms': list(pair), 'reason': reason.strip()})
    return {**core, 'method': 'sampled_joint_step_v2',
            'elbow_policy': {'mode': 'max_swivel', 'max_swivel_from_down_deg': float(threshold)},
            'extra_pair_exclusions': sorted(exclusions, key=lambda row: row['geoms'])}


def pair_policy_v2(checker, config):
    """Apply only the explicitly named v2 exclusions and retain an auditable pair table."""
    cfg = screening_config_v2(config)
    core = {key: cfg[key] for key in DEFAULT_SCREENING}
    core['method'] = 'sampled_joint_step_v1'
    margins, inherited = pair_policy(checker, core)
    pairs = checker.metadata['checked_pairs']
    pair_keys = [tuple(sorted(pair)) for pair in pairs]
    index_by_pair = {pair: index for index, pair in enumerate(pair_keys)}
    exclusion_map = {tuple(row['geoms']): row['reason'] for row in cfg['extra_pair_exclusions']}
    unknown = set(exclusion_map) - set(index_by_pair)
    if unknown:
        raise ValueError('显式排除对不存在或已被源碰撞策略排除: ' + repr(sorted(unknown)))
    retained = [index for index, pair in enumerate(pair_keys) if pair not in exclusion_map]
    if not retained:
        raise ValueError('v2碰撞检查对不能为空')
    table = []
    for index, pair in enumerate(pair_keys):
        reason = exclusion_map.get(pair)
        table.append({'geoms': list(pair), 'status': 'explicitly_excluded' if reason else 'checked',
                      'reason': reason, 'clearance_m': None if reason else float(margins[index])})
    policy = {
        'method': cfg['method'], 'collision_pairs': table,
        'checked_pairs': [table[index] for index in retained],
        'extra_pair_exclusions': cfg['extra_pair_exclusions'],
        'inherited_body_exclusions': inherited['explicit_body_exclusions'],
        'suppressed_pairs': inherited['suppressed_pairs'],
        'exclusion_provenance': 'named model exclusions plus only the frozen explicit v2 list; no automatic exclusions',
        'numeric_guard_m': NUMERIC_GUARD_M,
        'geometry_scope': 'MuJoCo compiled convex geometry',
        'scope': 'discrete approximation; no continuous collision or solver-substep guarantee',
    }
    return np.asarray(retained, dtype=np.int64), margins[retained], policy


def _screen_v2_state(q, checker, elbow, cfg, pair_indices, margins, policy):
    q = np.asarray(q)
    base = {'method': cfg['method'], 'max_joint_step_rad': cfg['max_joint_step_rad'],
            'numeric_guard_m': NUMERIC_GUARD_M, 'collision_margin_m': cfg['collision_margin_m'],
            'elbow_policy': cfg['elbow_policy'], 'scope': policy['scope']}

    def fail(reason, category, **detail):
        status = 'unknown' if category in ('inspection_incomplete', 'numerical_uncertainty') else 'rejected'
        return {**base, 'passed': False, 'status': status, 'reason': reason,
                'failure_category': category, **detail}

    if q.shape != (len(checker.jids),) or not np.isfinite(q).all():
        return fail('invalid_geometry_input', 'inspection_incomplete')
    low = np.array([checker.m.jnt_range[j, 0] for j in checker.jids])
    high = np.array([checker.m.jnt_range[j, 1] for j in checker.jids])
    if np.any(q < low) or np.any(q > high):
        return fail('joint_limits', 'confirmed_violation')
    distances = checker.distances(q)
    selected = distances[pair_indices].copy()
    if hasattr(checker, 'a'):
        import mujoco
        for local_index in np.flatnonzero((selected > 0) & (selected <= margins + NUMERIC_GUARD_M)):
            original_index = int(pair_indices[local_index])
            selected[local_index] = mujoco.mj_geomDistance(
                checker.m, checker.d, int(checker.a[original_index]), int(checker.b[original_index]), 10., None)
    if selected.shape != margins.shape or not np.isfinite(selected).all():
        return fail('collision_nonfinite', 'inspection_incomplete')
    gaps = selected - margins
    pair_index = int(np.argmin(gaps))
    if gaps[pair_index] <= NUMERIC_GUARD_M:
        pair = policy['checked_pairs'][pair_index]
        detail = {'limiting_pair': {'geoms': pair['geoms'], 'distance_lower_bound_m': float(selected[pair_index]),
                                    'clearance_m': float(margins[pair_index])}}
        if selected[pair_index] < -NUMERIC_GUARD_M:
            return fail('collision_penetration', 'confirmed_violation', **detail)
        return fail('collision_clearance_filter', 'clearance_filter', **detail)

    arm_reports = [elbow.metrics(points) for points in checker.d.xpos[elbow.bodies]]
    degenerate = []
    max_swivel = 0.0
    threshold = cfg['elbow_policy']['max_swivel_from_down_deg']
    for arm_index, report in enumerate(arm_reports):
        angle = report.get('swivel_from_down_deg')
        if angle is None or report.get('status') in ('degenerate_axis', 'degenerate_swivel'):
            degenerate.append(arm_index)
            continue
        if not math.isfinite(float(angle)):
            return fail('elbow_angle_nonfinite', 'inspection_incomplete', arms=arm_reports)
        angle = float(angle)
        max_swivel = max(max_swivel, angle)
        if abs(angle - threshold) <= V2_ELBOW_ANGLE_GUARD_DEG:
            return fail('elbow_threshold_guard', 'numerical_uncertainty', arms=arm_reports,
                        max_swivel_from_down_deg=max_swivel, angle_guard_deg=V2_ELBOW_ANGLE_GUARD_DEG)
        if angle > threshold:
            return fail('elbow_up', 'confirmed_violation', arms=arm_reports,
                        max_swivel_from_down_deg=max_swivel)
    return {**base, 'passed': True, 'status': 'accepted', 'arms': arm_reports,
            'degenerate_arm_count': len(degenerate), 'degenerate_arms': degenerate,
            'max_swivel_from_down_deg': max_swivel}


def screen_polyline_v2(q, checker, elbow, config=None, policy=None):
    cfg = screening_config_v2(config)
    if policy is None:
        pair_indices, margins, pair_table = pair_policy_v2(checker, cfg)
    else:
        pair_indices, margins, pair_table = policy
    q = np.asarray(q)
    base = {'method': cfg['method'], 'max_joint_step_rad': cfg['max_joint_step_rad'],
            'numeric_guard_m': NUMERIC_GUARD_M, 'collision_margin_m': cfg['collision_margin_m'],
            'elbow_policy': cfg['elbow_policy'], 'extra_pair_exclusions': cfg['extra_pair_exclusions'],
            'collision_pairs': pair_table['collision_pairs'], 'scope': pair_table['scope']}

    def fail(reason, category, **detail):
        status = 'unknown' if category in ('inspection_incomplete', 'numerical_uncertainty') else 'rejected'
        return {**base, 'passed': False, 'status': status, 'reason': reason,
                'failure_category': category, **detail}

    if q.ndim != 2 or q.shape[1] != len(checker.jids) or len(q) < 2 or not np.isfinite(q).all():
        return fail('invalid_geometry_input', 'inspection_incomplete')
    low = np.array([checker.m.jnt_range[j, 0] for j in checker.jids])
    high = np.array([checker.m.jnt_range[j, 1] for j in checker.jids])
    invalid = np.flatnonzero(np.any((q < low[None, :]) | (q > high[None, :]), axis=1))
    if len(invalid):
        return fail('joint_limits', 'confirmed_violation', source_frame=int(invalid[0]), checked_samples=0)
    steps = np.maximum(1, np.ceil(np.max(np.abs(np.diff(q, axis=0)), axis=1) /
                                  cfg['max_joint_step_rad']).astype(np.int64))
    needed = 1 + sum(int(n) for n in steps)
    if needed > cfg['max_samples']:
        return fail('sample_budget', 'inspection_incomplete', required_samples=needed, checked_samples=0)
    checked = degenerate_arm_samples = 0
    max_swivel = 0.0
    for frame, state in [(0, q[0])]:
        report = _screen_v2_state(state, checker, elbow, cfg, pair_indices, margins, pair_table)
        checked += 1
        if not report['passed']:
            return fail(report['reason'], report['failure_category'], segment=frame, fraction=0.,
                        checked_samples=checked, required_samples=needed, sample_report=report)
        degenerate_arm_samples += report['degenerate_arm_count']
        max_swivel = max(max_swivel, report['max_swivel_from_down_deg'])
    for frame, count in enumerate(steps):
        for k in range(1, int(count) + 1):
            fraction = k / int(count)
            state = q[frame + 1] if k == count else q[frame] + fraction * (q[frame + 1] - q[frame])
            report = _screen_v2_state(state, checker, elbow, cfg, pair_indices, margins, pair_table)
            checked += 1
            if not report['passed']:
                return fail(report['reason'], report['failure_category'], segment=frame,
                            fraction=fraction, checked_samples=checked, required_samples=needed,
                            sample_report=report)
            degenerate_arm_samples += report['degenerate_arm_count']
            max_swivel = max(max_swivel, report['max_swivel_from_down_deg'])
    return {**base, 'passed': True, 'status': 'accepted', 'source_frames': len(q),
            'checked_samples': checked, 'required_samples': needed, 'degenerate_arm_samples': degenerate_arm_samples,
            'max_swivel_from_down_deg': max_swivel}


def screen_track_windows_v2(q, checker, elbow, config=None, policy=None):
    """Return a conservative frame mask, invalidating both ends of a failing edge."""
    cfg = screening_config_v2(config)
    if policy is None:
        pair_indices, margins, pair_table = pair_policy_v2(checker, cfg)
    else:
        pair_indices, margins, pair_table = policy
    q = np.asarray(q)
    if q.ndim != 2 or q.shape[1] != len(checker.jids) or len(q) < 2 or not np.isfinite(q).all():
        return {'status': 'unknown', 'valid_frames': np.zeros(len(q), dtype=bool),
                'failure_counts': {'invalid_geometry_input': 1}, 'checked_samples': 0}
    steps = np.maximum(1, np.ceil(np.max(np.abs(np.diff(q, axis=0)), axis=1) /
                                  cfg['max_joint_step_rad']).astype(np.int64))
    needed = len(q) + sum(int(n) - 1 for n in steps)
    if needed > cfg['max_samples']:
        return {'status': 'unknown', 'valid_frames': np.zeros(len(q), dtype=bool),
                'failure_counts': {'sample_budget': 1}, 'required_samples': needed, 'checked_samples': 0}
    valid = np.ones(len(q), dtype=bool)
    failures = Counter()
    checked = degenerate = 0

    def inspect(state, frame_indices):
        nonlocal checked, degenerate
        report = _screen_v2_state(state, checker, elbow, cfg, pair_indices, margins, pair_table)
        checked += 1
        if report['passed']:
            degenerate += report['degenerate_arm_count']
            return
        failures[report['reason']] += 1
        for index in frame_indices:
            valid[index] = False

    for i, state in enumerate(q):
        inspect(state, (i,))
    for frame, count in enumerate(steps):
        for k in range(1, int(count)):
            fraction = k / int(count)
            inspect(q[frame] + fraction * (q[frame + 1] - q[frame]), (frame, frame + 1))
    return {'status': 'accepted' if valid.all() else 'partial' if valid.any() else 'rejected',
            'valid_frames': valid, 'failure_counts': dict(failures), 'checked_samples': checked,
            'required_samples': needed, 'degenerate_arm_samples': degenerate,
            'invalid_frame_count': int((~valid).sum()), 'collision_pairs': pair_table['collision_pairs'],
            'extra_pair_exclusions': cfg['extra_pair_exclusions']}
