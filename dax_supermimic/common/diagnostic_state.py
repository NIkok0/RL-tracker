"""Restore an independent snapshot without aliasing optimizer tensors."""
import copy


def restore_diagnostic_state(objects, snapshot):
    for key, value in snapshot.items():
        objects[key].load_state_dict(copy.deepcopy(value))
