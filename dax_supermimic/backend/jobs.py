"""Local background generation process lifecycle; no HTTP parameter parsing."""
import json
import subprocess
import sys
from threading import Lock
from ..common.paths import ROOT

class GenerationJobs:
    def __init__(self,pipeline,simulation_running=lambda:False):
        self.pipeline=pipeline
        self.simulation_running=simulation_running
        self.lock=Lock()
        self.process=None

    def start(self,payload):
        if set(payload)!={'job_id'}:raise ValueError('只接受已冻结的任务ID；修改参数请新建任务')
        with self.lock:
            if self.process is not None and self.process.poll()is None:
                raise ValueError('已有数据生成任务正在运行')
            if self.simulation_running():raise ValueError('请先关闭策略仿真，再启动数据生成')
            job_id=payload['job_id'];folder=self.pipeline._dir(job_id)
            state=self.pipeline.status(job_id)
            if state['status']=='published':raise ValueError('该任务已发布，无需重复执行')
            plan=json.loads((folder/'plan.json').read_text())
            self.pipeline._check_bindings(plan)
            if state['plan_sha256']!=__import__('hashlib').sha256((folder/'plan.json').read_bytes()).hexdigest():raise ValueError('冻结计划被修改')
            # Never pass executable paths, command text or parameter overrides from HTTP.
            command=[sys.executable,str(ROOT/'scripts/run_dax_augmentation.py'),'--job_id',job_id,'--workers','6']
            with (folder/'runner.log').open('ab')as log:
                process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            self.process=process
            return {**self.pipeline.status(job_id),'execution_requested':True,'runner_pid':process.pid,'runner_log':str(folder/'runner.log')}
