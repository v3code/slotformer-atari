from .obj3d import build_obj3d_dataset, build_obj3d_slots_dataset
from .clevrer import build_clevrer_dataset, build_clevrer_slots_dataset
from .physion import build_physion_dataset, build_physion_slots_dataset, \
    build_physion_slots_label_dataset
from .phyre import build_phyre_dataset, build_phyre_slots_dataset, \
    build_phyre_rollout_slots_dataset
from .pong import build_pong_dataset, build_pong_slots_dataset
from .spinv import build_spinv_slots_dataset, build_spinv_dataset
from .crafter import build_crafter_dataset, build_crafter_slots_dataset
from .cubes import build_cubes_dataset, build_cubes_slots_dataset
from .shapes import build_shapes_dataset, build_shapes_slots_dataset
from .nav import build_navigation_dataset, build_navigation_slots_dataset

def build_dataset(params, val_only=False):
    dst = params.dataset
    if 'physion' not in dst:
        return eval(f'build_{dst}_dataset')(params, val_only=val_only)
    # physion dataset looks like 'physion_xxx_$SUBSET'
    return eval(f"build_{dst[:dst.rindex('_')]}_dataset")(
        params, val_only=val_only)
