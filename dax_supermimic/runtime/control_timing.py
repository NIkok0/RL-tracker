"""Bounded soft timing and transport backpressure, no hardware side effects."""
import math
import os
import threading
import time


def pin_control_thread(cpu):
    """Pin only the caller, after background transport threads have started."""
    before=sorted(os.sched_getaffinity(0))
    if cpu is not None:
        if cpu not in before:
            raise ValueError(f'CPU {cpu} is not in the allowed affinity set {before}')
        os.sched_setaffinity(threading.get_native_id(),{cpu})
    return {'thread_id':threading.get_native_id(),'allowed_before':before,
            'allowed_after':sorted(os.sched_getaffinity(0))}


class StartPacer:
    """Pace publish starts; rebase on lateness, never catch up with bursts."""
    def __init__(self, interval_s, clock=time.monotonic, sleep=time.sleep):
        if not math.isfinite(interval_s) or interval_s<=0:
            raise ValueError('Positive finite interval required')
        self.interval_s=interval_s;self.clock=clock;self.sleep=sleep
        self.last_start=None

    def wait(self):
        if self.last_start is not None:
            remaining=self.last_start+self.interval_s-self.clock()
            if remaining>0:self.sleep(remaining)

    def started(self):
        now=self.clock()
        elapsed=None if self.last_start is None else now-self.last_start
        self.last_start=now
        return elapsed


def publish_with_backpressure(client,topic,msg_type,message,*,codec,max_wait_s=.02,
                              ttl_s=.04,clock=time.monotonic,sleep=time.sleep,
                              queue_full_error=RuntimeError):
    """Retry only an explicit rejection before enqueue, within a fixed TTL.

    Other failures may have ambiguous delivery and must never be retried here.
    A successful return still confirms enqueue only, not actuator application.
    """
    if not all(math.isfinite(x) and x>=0 for x in (max_wait_s,ttl_s)) or ttl_s<=max_wait_s:
        raise ValueError('Require 0 <= wait < positive target TTL')
    start=clock();end=start+max_wait_s;deadline=start+ttl_s;retries=0
    while True:
        if not client.is_connected:raise RuntimeError('Writer disconnected')
        try:
            frame_id=client.publish(topic,msg_type,message,codec=codec,deadline=deadline)
            return frame_id,{'wait_s':clock()-start,'queue_full_retries':retries,'deadline':deadline}
        except queue_full_error as exc:
            if str(exc)!='outgoing queue full':raise
            if clock()>=end:raise TimeoutError('Publish queue wait budget exhausted') from exc
            retries+=1;sleep(min(.0005,max(0,end-clock())))


class ArmWriters:
    """Independent normal/hold streams, exact per-arm enqueue bookkeeping."""
    def __init__(self,writers,hold_writers,topics,msg_type,make_message,codec,
                 queue_full_error,max_wait_s=.02,ttl_s=.04):
        self.writers=writers;self.hold_writers=hold_writers;self.topics=topics
        self.msg_type=msg_type;self.make_message=make_message;self.codec=codec
        self.queue_full_error=queue_full_error;self.max_wait_s=max_wait_s;self.ttl_s=ttl_s
        self.last_accepted={}

    def send(self,targets,emit):
        frames={}
        # Caller validates both arms before this method. Capture a partial pair
        # immediately so abort holds the actual last accepted target per arm.
        for side,target in targets.items():
            frame,stats=publish_with_backpressure(self.writers[side],self.topics[side],
                self.msg_type,self.make_message(target),codec=self.codec,
                max_wait_s=self.max_wait_s,ttl_s=self.ttl_s,queue_full_error=self.queue_full_error)
            self.last_accepted[side]=list(target);frames[side]=frame
            emit(side,target,frame,stats)
        return frames

    def hold(self,emit):
        # Independent connections keep ordinary queue congestion from preventing
        # every hold attempt. Failure is still possible and explicitly reported.
        for side,target in self.last_accepted.items():
            try:
                frame,stats=publish_with_backpressure(self.hold_writers[side],self.topics[side],
                    self.msg_type,self.make_message(target),codec=self.codec,
                    max_wait_s=self.max_wait_s,ttl_s=self.ttl_s,queue_full_error=self.queue_full_error)
                emit(side,target,frame,stats,None)
            except Exception as exc:
                emit(side,target,None,None,str(exc))
