import enum
import os


class Environments(str, enum.Enum):
    PONG = "pong"
    SPACE_INVADERS = "spinv"
    CRAFTER = "crafter"
    CUBES_3D = 'cubes'
    NAVIGATION_5x5 = "nav5x5"
    PUSHING_5x5 = "push5x5"
    SHAPES_2D = 'shapes'


STATE_FOLDER_TEMPLATE = "e_{:d}"
ACTIONS_TEMPLATE = "actions.npy"
ACTIONS_FOLDER_TEMPLATE = os.path.join(STATE_FOLDER_TEMPLATE, ACTIONS_TEMPLATE)
STATE_IDS_TEMPLATE = "state_ids.npy"
STATE_IDS_FOLDER_TEMPLATE = os.path.join(STATE_FOLDER_TEMPLATE, STATE_IDS_TEMPLATE)
STATE_TEMPLATE = os.path.join(STATE_FOLDER_TEMPLATE, "s_t_{:d}.png")
