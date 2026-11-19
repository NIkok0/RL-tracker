"""Optional W&B telemetry. No model watching, checkpoint uploads or training changes."""
from __future__ import annotations

import atexit
from contextlib import contextmanager
import importlib
import json
import math
import os
from pathlib import Path
import random
import sys
import uuid
import warnings


def add_wandb_args(parser):
    group = parser.add_argument_group('Weights & Biases monitoring')
    group.add_argument('--wandb_mode', choices=('disabled', 'offline', 'online'), default='disabled')
    group.add_argument('--wandb_project', default='dax-supermimic')
    group.add_argument('--wandb_entity', default=None)
    group.add_argument('--wandb_name', default=None)
    group.add_argument('--wandb_group', default=None)
    group.add_argument('--wandb_tags', nargs='*', default=[])
    group.add_argument('--wandb_run_id', default=None, help='Explicit W&B run identity; otherwise persisted per log_dir')


@contextmanager
def preserve_rng():
    """SDK initialization/logging must not advance the trainer's RNG streams."""
    python_state = random.getstate()
    np = sys.modules.get('numpy')
    numpy_state = np.random.get_state() if np is not None else None
    torch = sys.modules.get('torch')
    cpu_state = torch.random.get_rng_state() if torch is not None else None
    cuda_state = torch.cuda.get_rng_state() if torch is not None and torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        if numpy_state is not None: np.random.set_state(numpy_state)
        if cpu_state is not None: torch.random.set_rng_state(cpu_state)
        if cuda_state is not None: torch.cuda.set_rng_state(cuda_state)


class WandbMonitor:
    def __init__(self, args, start_iteration=0):
        self.args = args
        self.rank = int(os.environ.get('RANK', '0'))
        self.world = int(os.environ.get('WORLD_SIZE', '1'))
        self.mode = getattr(args, 'wandb_mode', 'disabled')
        self.requested = self.mode != 'disabled'
        self.run = None
        self.closed = False
        self.iteration = int(start_iteration)
        self.path = Path(args.log_dir) / 'wandb_monitor.json'
        self.identity = {}
        if not self.requested or self.rank:
            return
        try:
            old = json.loads(self.path.read_text()) if self.path.exists() else {}
            same_project = old.get('project') == args.wandb_project and old.get('entity') == args.wandb_entity
            reuse = bool(getattr(args, 'resume', None) and same_project and old.get('last_iteration', 0) <= start_iteration)
            run_id = args.wandb_run_id or (old.get('run_id') if reuse else None) or uuid.uuid4().hex[:12]
            self.identity = {'run_id': run_id, 'mode': self.mode, 'project': args.wandb_project,
                             'entity': args.wandb_entity, 'start_iteration': start_iteration,
                             'source_checkpoint': getattr(args, 'resume', None), 'world_size': self.world}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with preserve_rng():
                sdk = importlib.import_module('wandb')
                # Explicit config allowlist avoids copying environment credentials or arbitrary CLI extensions.
                keys = ('num_envs', 'global_num_envs', 'iterations', 'buffer_steps', 'utd', 'horizon',
                        'goal_frames', 'hist', 'wm_window', 'wm_batches', 'wm_batch', 'global_wm_batch',
                        'pi_batch', 'global_pi_batch', 'lr', 'gamma', 'lam', 'action_noise', 'critic_epochs',
                        'critic_minibatches', 'no_critic', 'warmup_steps', 'wm_only_iters', 'lr_decay_iters',
                        'ckpt_every', 'log_every', 'motion_dir', 'seed', 'amp', 'geometry_asset', 'data_manifest')
                config = {key: getattr(args, key) for key in keys if hasattr(args, key)}
                config.update(world_size=self.world, resumed_from_iteration=start_iteration,
                              source_checkpoint=getattr(args, 'resume', None))
                self.run = sdk.init(project=args.wandb_project, entity=args.wandb_entity,
                     name=args.wandb_name or Path(args.log_dir).name, group=args.wandb_group,
                     tags=args.wandb_tags, id=run_id, mode=self.mode,
                     resume='allow' if self.mode == 'online' and (reuse or args.wandb_run_id) else None,
                     dir=str(self.path.parent), config=config,
                     settings=sdk.Settings(disable_git=True, save_code=False, console='off', init_timeout=15))
                self.run.define_metric('train/iteration')
                self.run.define_metric('train/*', step_metric='train/iteration')
                self.run.define_metric('progress/*', step_metric='train/iteration')
                self.run.define_metric('eval/iteration')
                self.run.define_metric('eval/*', step_metric='eval/iteration')
                self.identity['run_url'] = getattr(self.run, 'url', None)
            self._state('running')
            atexit.register(self.finish, exit_code=1)
        except Exception as error:
            self._failed(error)

    def _state(self, state, **extra):
        value = {**self.identity, 'state': state, 'last_iteration': self.iteration, **extra}
        temp = self.path.with_suffix('.json.tmp')
        temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
        temp.replace(self.path)

    def _failed(self, error):
        # Do not echo SDK exceptions that could contain account URLs or credentials.
        warnings.warn('W&B monitoring failed (' + type(error).__name__ + '); local training/CSV continue.', RuntimeWarning)
        try: self._state('monitor_error', error_type=type(error).__name__)
        except OSError: pass
        if self.run is not None:
            try: self.run.finish(exit_code=1)
            except Exception: pass
        self.run = None

    def _log(self, metrics):
        if self.run is None or self.closed: return
        try:
            with preserve_rng(): self.run.log(metrics)
            self._state('running')
        except Exception as error: self._failed(error)

    def log_training(self, row, agg, stats):
        if not self.requested: return
        # Aggregate on every rank even if rank0's SDK failed: no asymmetric collectives.
        # Episode fields are upstream means over vectorized collection batches, not episode-weighted means.
        specs = [('wm_loss', 3, agg['wm']), ('wm_error_last_frame', 4, agg['wm8']),
                 ('actor_loss', 5, agg['pi']), ('critic_loss', 6, agg['v']),
                 ('imagined_reward', 7, agg['ir']), ('actor_grad_norm', 8, agg['gn']),
                 ('imagined_survival', 9, agg['surv']), ('sim_reward', 10, stats['r']),
                 ('episode_length_batch_mean', 11, stats['ep_len']),
                 ('episode_return_batch_mean', 12, stats['ep_ret'])]
        pairs = [[float(row[index])*len(values), float(len(values))] for _, index, values in specs]
        pairs.append([sum(float(x) for x in stats['term']), sum(float(x) for x in stats['done'])])
        if self.world > 1:
            import torch
            import torch.distributed as dist
            packed = torch.tensor(pairs, dtype=torch.float64,
                                  device='cuda:0' if dist.get_backend() == 'nccl' else 'cpu')
            dist.all_reduce(packed)
            pairs = packed.cpu().tolist()
        metrics = {f'train/{name}': total/count for (name, _, _), (total, count) in zip(specs, pairs) if count}
        if pairs[-1][1]: metrics['train/termination_fraction'] = pairs[-1][0]/pairs[-1][1]
        metrics.update({'train/iteration': int(row[0]), 'progress/wall_hours': float(row[1]),
                        'progress/sim_steps': int(row[2]), 'train/learning_rate': float(row[15]),
                        'progress/iterations_per_second_rank0': float(row[14])})
        self.iteration = int(row[0])
        if self.rank == 0: self._log(metrics)

    def log_evaluation(self, directory, iteration):
        """Publish completed result files, never an in-progress or timed-out evaluation."""
        if self.rank or self.run is None or self.closed: return
        try:
            directory = Path(directory)
            result = json.loads((directory/'result.json').read_text())
            metrics = {'eval/iteration': int(iteration)}
            for key in ('norm_return', 'norm_return_framew', 'n_failed'):
                if key in result: metrics['eval/'+key] = result[key]
            if (directory/'tracking.json').exists():
                tracking = json.loads((directory/'tracking.json').read_text())['per_clip']
                metrics['eval/clip_count'] = len(tracking)
                for key in ('joint_rmse_rad', 'left_wrist_rmse_m', 'right_wrist_rmse_m',
                            'left_wrist_p95_m', 'right_wrist_p95_m'):
                    values = [float(row[key]) for row in tracking.values()]
                    if values: metrics['eval/'+key+'_clip_mean'] = sum(values)/len(values)
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in metrics.values()):
                raise ValueError('Nonfinite evaluation metrics')
            self._log(metrics)
        except Exception as error:
            warnings.warn('W&B evaluation metrics skipped ('+type(error).__name__+'); see local evaluation files.', RuntimeWarning)

    def finish(self, exit_code=0):
        if self.closed: return
        self.closed = True
        if self.run is not None:
            try:
                with preserve_rng(): self.run.finish(exit_code=exit_code)
                self._state('finished' if exit_code == 0 else 'interrupted')
            except Exception as error: self._failed(error)

    def finish_training(self):
        # Bounded runners still need to publish their final evaluation after the loop exits.
        if not getattr(self, 'defer_finish', False): self.finish()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Publish an existing completed snapshot evaluation to W&B')
    parser.add_argument('--log_dir', required=True)
    parser.add_argument('--evaluation_dir', required=True)
    parser.add_argument('--iteration', type=int, required=True)
    parser.add_argument('--resume', default='existing-evaluation')
    add_wandb_args(parser)
    args = parser.parse_args()
    monitor = WandbMonitor(args, start_iteration=args.iteration)
    monitor.log_evaluation(args.evaluation_dir, args.iteration)
    monitor.finish()


if __name__ == '__main__': main()
