import enum
import os


class Environments(str, enum.Enum):
    PONG = "pong"
    SPACE_INVADERS = "space-invaders"


STATE_FOLDER_TEMPLATE = "e_{:d}"
ACTIONS_TEMPLATE = os.path.join(STATE_FOLDER_TEMPLATE,"actions.pkl")
STATE_IDS_TEMPLATE = os.path.join(STATE_FOLDER_TEMPLATE,"state_ids.pkl")
STATE_TEMPLATE = os.path.join(STATE_FOLDER_TEMPLATE, "s_t_{:d}.png")
