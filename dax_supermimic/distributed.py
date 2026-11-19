"""Synchronous data parallel primitives; averages gradients before original clipping.

Each process sees exactly one GPU. Global batch/environment counts are sharded,
normalizer moments are merged before any consumer, optimizer states stay replicated.
"""
from __future__ import annotations
import hashlib,json,os,time,signal,subprocess,sys
from pathlib import Path
from datetime import timedelta
import torch
import torch.distributed as dist

class Finished(BaseException): pass

class DistributedTraining:
    def __init__(self, budget_seconds, smoke_iterations, output):
        self.rank=int(os.environ['RANK']);self.world=int(os.environ['WORLD_SIZE'])
        dist.init_process_group('nccl',timeout=timedelta(minutes=30))
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.budget=budget_seconds;self.smoke=smoke_iterations;self.started=time.time()
        # float64 timestamps avoid loss of seconds at Unix epoch magnitude.
        timestamp=torch.tensor(self.started,dtype=torch.float64,device='cuda');dist.broadcast(timestamp,0)
        self.started=timestamp.item();self.g=None;self.stop=False;self.mid=False
        self.counts={'wm':0,'actor':0,'critic':0};self.last_status=0;self.sync_calls=0
        signal.signal(signal.SIGTERM,lambda *_:setattr(self,'stop',True))

    def attach(self,g):
        self.g=g;self.start_it=g['start_it'];self.inherited=g['sim_steps']
        g['monitor'].defer_finish=True
        self.resume_check()
        for name,key in [('wm_update','wm'),('policy_update','actor')]:
            original=g[name]
            def checked(*a,_original=original,_key=key,**kw):
                result=_original(*a,**kw)
                ok=torch.tensor(int(all(bool(torch.isfinite(x)) for x in result)),device='cuda')
                dist.all_reduce(ok,op=dist.ReduceOp.MIN)
                if not ok.item():raise RuntimeError('Nonfinite '+_key+' output on at least one rank')
                self.counts[_key]+=1
                if _key=='actor':self.counts['critic']+=g['args'].critic_epochs*g['args'].critic_minibatches
                return result
            g[name]=checked
        torch.manual_seed(g['args'].seed+self.rank)
        self.check_equal('initial')

    def resume_check(self):
        g=self.g
        if not g['args'].resume:raise ValueError('Distributed admission requires an immutable resume checkpoint')
        ck=torch.load(g['args'].resume,map_location='cuda:0',weights_only=False)
        def equal(a,b):
            if torch.is_tensor(a):return torch.equal(a,b)
            if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
            if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
            return a==b
        for name in ('wm','pi','critic','opt_wm','opt_pi','opt_v'):
            if not equal(g[name].state_dict(),ck[name]):raise RuntimeError('Resume mismatch: '+name)
        if g['start_it']!=ck['iteration'] or g['sim_steps']!=ck['sim_steps']:raise RuntimeError('Resume counter mismatch')
        del ck

    def clip(self,parameters,max_norm,**kwargs):
        params=list(parameters)
        present=torch.tensor([int(p.grad is not None) for p in params],device='cuda')
        low=present.clone();high=present.clone();dist.all_reduce(low,op=dist.ReduceOp.MIN);dist.all_reduce(high,op=dist.ReduceOp.MAX)
        if not torch.equal(low,high):raise RuntimeError('Gradient presence differs between ranks')
        grads=[p.grad for p in params if p.grad is not None]
        if grads:
            flat=torch.cat([x.reshape(-1) for x in grads]);dist.all_reduce(flat);flat.div_(self.world)
            if not bool(torch.isfinite(flat).all()):raise RuntimeError('Nonfinite synchronized gradients')
            offset=0
            for grad in grads:grad.copy_(flat[offset:offset+grad.numel()].view_as(grad));offset+=grad.numel()
        self.sync_calls+=1
        return torch.nn.utils.clip_grad_norm_(params,max_norm,**kwargs)

    @torch.no_grad()
    def norm_update(self,norm,x):
        x=x.reshape(-1,x.shape[-1]).float()
        if norm.count>=norm.max_count:return
        n=torch.tensor(float(len(x)),device=x.device,dtype=torch.float64)
        total=x.double().sum(0);squares=x.double().square().sum(0)
        packed=torch.cat((n[None],total,squares));dist.all_reduce(packed)
        if not bool(torch.isfinite(packed).all()) or packed[0]<=0:raise RuntimeError('Invalid global normalization moments')
        n=packed[0].float();mean64=packed[1:1+x.shape[-1]]/packed[0];m=mean64.float()
        v=(packed[1+x.shape[-1]:]/packed[0]-mean64.square()).clamp_min(0).float()
        count=norm.count+n;delta=m-norm.mean
        norm.mean.add_(delta*n/count)
        norm.var.copy_((norm.var*norm.count+v*n+delta.square()*norm.count*n/count)/count)
        norm.count.copy_(count)

    def digest(self):
        h=hashlib.sha256()
        def add(x):
            if torch.is_tensor(x):h.update(str((tuple(x.shape),x.dtype)).encode());h.update(x.detach().cpu().contiguous().numpy().tobytes())
            elif isinstance(x,dict):
                for k,v in sorted(x.items(),key=lambda kv:str(kv[0])):h.update(str(k).encode());add(v)
            elif isinstance(x,(list,tuple)):
                for v in x:add(v)
            else:h.update(repr(x).encode())
        for name in ('wm','pi','critic','opt_wm','opt_pi','opt_v'):add(self.g[name].state_dict())
        return h.hexdigest()

    def check_equal(self,label):
        digest=self.digest();out=[None]*self.world;dist.all_gather_object(out,digest)
        if len(set(out))!=1:raise RuntimeError('Model/normalizer/optimizer mismatch across ranks: '+label)
        if label=='initial':self.initial_digest=digest
        if self.rank==0:self.write('sync_'+label+'.json',{'passed':True,'world_size':self.world,'digests':out,'gradient_sync_calls_per_rank':self.sync_calls,'changed_since_resume':digest!=getattr(self,'initial_digest',digest)})

    def write(self,name,value):
        path=self.output/name;tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

    def status(self,state,iteration):
        if self.rank==0:self.write('job_status.json',{'state':state,'pid':os.getpid(),'world_size':self.world,'iteration':iteration,'start_iteration':self.start_it,'elapsed_s':time.time()-self.started,'budget_seconds':self.budget,'deadline_unix_s':self.started+self.budget,'sim_steps':self.g['sim_steps'],'new_sim_steps':self.g['sim_steps']-self.inherited,'synchronized_updates':self.counts,'global_envs':self.g['args'].global_num_envs,'global_wm_batch':self.g['args'].global_wm_batch,'global_pi_batch':self.g['args'].global_pi_batch})

    def save(self,name,iteration):
        g=self.g
        if self.rank==0:
            ck={n:g[n].state_dict() for n in ('wm','pi','critic','opt_wm','opt_pi','opt_v')}
            from .actions import runtime_action_metadata
            ck.update(iteration=iteration,wall_time=g['wall_offset']+time.time()-self.started,sim_steps=g['sim_steps'],args=vars(g['args']),action_contract=g['ACTION_CONTRACT_ID'],runtime_action=runtime_action_metadata(),distributed_world_size=self.world,source_checkpoint=g['args'].resume)
            path=self.output/name;torch.save(ck,str(path)+'.tmp');os.replace(str(path)+'.tmp',path)
        dist.barrier()

    def evaluate(self,name,label):
        if self.rank==0:
            remaining=self.started+self.budget-time.time()
            if remaining>30:
                with (self.output/(label+'.log')).open('w') as f:
                    try:
                        result=subprocess.run([sys.executable,'scripts/dax_snapshot_evaluate.py','--checkpoint',str(self.output/name),'--out',str(self.output/label)],stdout=f,stderr=subprocess.STDOUT,timeout=max(1,min(900,remaining-5)))
                        self.write(label+'_exit.json',{'returncode':result.returncode})
                        if result.returncode==0 and 'monitor' in self.g:
                            self.g['monitor'].log_evaluation(self.output/label,self.g['it'])
                    except subprocess.TimeoutExpired:self.write(label+'_exit.json',{'returncode':'timeout'})
        dist.barrier()

    def boundary(self,iteration):
        flags=torch.tensor([int(self.stop),int(time.time()-self.started>=self.budget-900),int(bool(self.smoke) and iteration-self.start_it>=self.smoke)],device='cuda')
        dist.all_reduce(flags,op=dist.ReduceOp.MAX)
        if bool(flags.any()):
            self.check_equal('final');self.save('budget_final.pt',iteration)
            if not self.smoke:self.evaluate('budget_final.pt','evaluation_final')
            self.status('completed_smoke' if self.smoke else 'completed_budget',iteration)
            raise Finished()
        # All ranks use rank0's midpoint decision.
        mid=torch.tensor(int(not self.smoke and not self.mid and time.time()-self.started>=self.budget/2) if self.rank==0 else 0,device='cuda');dist.broadcast(mid,0)
        if mid.item():
            self.check_equal('mid');self.save('midpoint.pt',iteration);self.evaluate('midpoint.pt','evaluation_midpoint');self.mid=True
        if iteration>self.start_it and iteration%500==0:
            self.check_equal('periodic');self.save('last.pt',iteration)
        if time.time()-self.last_status>20:self.status('training',iteration);self.last_status=time.time()
