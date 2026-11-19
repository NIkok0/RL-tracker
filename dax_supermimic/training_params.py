"""Optional user hyperparameters, separate from versioned training job bindings."""

import json
import math
from pathlib import Path


POSITIVE_INTS = set('num_envs iterations buffer_steps horizon goal_frames hist wm_window wm_batches wm_batch pi_batch critic_epochs critic_minibatches warmup_steps lr_decay_iters ckpt_every log_every'.split())
NONNEGATIVE_INTS = {'seed', 'wm_only_iters'}
NUMBERS = {'lr', 'utd', 'gamma', 'lam', 'action_noise', 'policy_ref_anchor_weight'}
CHOICES = {'amp': {'none', 'bf16'}, 'reset_sampling_mode': {'original', 'uniform'}}
TRAIN_KEYS = POSITIVE_INTS | NONNEGATIVE_INTS | NUMBERS | CHOICES.keys() | {'no_critic'}
BUDGET_KEYS = {'budget_seconds', 'evaluation_at_seconds', 'final_evaluation_reserve_seconds'}
REWARD_SCALES = set('body_pos_std body_ori_std body_linvel_std body_angvel_std root_pos_std root_ori_std'.split())
REWARD_WEIGHTS = {'w_action_rate', 'w_joint_limit'}


def _mapping(value, name, allowed):
    if not isinstance(value, dict):
        raise ValueError(f'{name} must be a mapping')
    unknown = value.keys() - allowed
    if unknown:
        raise ValueError(f'{name}: unknown keys: {", ".join(sorted(map(str, unknown)))}')
    return value


def validate_reward(value):
    reward = _mapping(value, 'reward', REWARD_SCALES | REWARD_WEIGHTS)
    for key, number in reward.items():
        if type(number) not in (int, float) or not math.isfinite(number):
            raise ValueError(f'reward.{key} must be a finite number')
        if key in REWARD_SCALES and number <= 0:
            raise ValueError(f'reward.{key} must be positive')
        if key in REWARD_WEIGHTS and number > 0:
            raise ValueError(f'reward.{key} must be nonpositive')
    return reward


def load_training_params(path: Path, preset: str):
    import yaml

    class UniqueKeyLoader(yaml.SafeLoader):
        def construct_mapping(self, node, deep=False):
            pairs = self.construct_pairs(node, deep=deep)
            result = {}
            for key, value in pairs:
                if not isinstance(key, str) or key in result:
                    raise ValueError(f'Invalid or duplicate YAML key: {key!r}')
                result[key] = value
            return result

    raw = _mapping(yaml.load(path.read_text(encoding='utf-8'), Loader=UniqueKeyLoader),
                   'YAML', {'training', 'reward', 'local', 't14'})
    shared = _mapping(raw.get('training', {}), 'training', TRAIN_KEYS)
    local = _mapping(raw.get('local', {}), 'local', TRAIN_KEYS | BUDGET_KEYS)
    t14 = _mapping(raw.get('t14', {}), 't14', TRAIN_KEYS)
    reward = validate_reward(raw.get('reward', {}))
    for section, values in (('training', shared), ('local', local), ('t14', t14)):
        for key, value in values.items():
            if key in POSITIVE_INTS | BUDGET_KEYS:
                valid = type(value) is int and value >= (0 if key == 'evaluation_at_seconds' else 1)
            elif key in NONNEGATIVE_INTS:
                valid = type(value) is int and value >= 0
            elif key in NUMBERS:
                valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
                if valid and key in {'lr', 'utd'}:
                    valid = value > 0
                if valid and key in {'gamma', 'lam'}:
                    valid = value <= 1 and (key != 'gamma' or value < 1)
            elif key in CHOICES:
                valid = isinstance(value, str) and value in CHOICES[key]
            else:
                valid = type(value) is bool
            if not valid:
                raise ValueError(f'{section}.{key}: invalid value {value!r}')
    selected = {'local': local, 't14': t14}[preset]
    budget = {key: value for key, value in selected.items() if key in BUDGET_KEYS}
    training = {**shared, **{key: value for key, value in selected.items() if key in TRAIN_KEYS}}
    for key in ('buffer_steps', 'warmup_steps'):
        if key in training and 'wm_window' in training and training[key] < training['wm_window']:
            raise ValueError(f'{key} must be at least wm_window')
    if not training.get('no_critic', False) and {'horizon', 'pi_batch', 'critic_minibatches'} <= training.keys():
        if training['critic_minibatches'] > training['horizon'] * training['pi_batch']:
            raise ValueError('critic_minibatches exceeds the number of imagined transitions')
    return training, reward, budget


def overlay_trainer_args(base, training, reward):
    """Replace existing option values and append options absent from the launcher."""
    args = list(base)
    for key, value in training.items():
        flag = '--' + key
        if key == 'no_critic':
            if value:
                args.append(flag)
        elif flag in args:
            args[args.index(flag) + 1] = str(value)
        else:
            args.extend([flag, str(value)])
    if reward:
        args.extend(['--reward_overrides', json.dumps(reward, allow_nan=False)])
    return args
