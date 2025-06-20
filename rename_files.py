import os
from Constants import *

for i, file in enumerate(os.listdir(PATH_SAVE_CALIBRE_PROJECTOR)):
    os.rename(os.path.join(PATH_SAVE_CALIBRE_PROJECTOR, file), os.path.join(PATH_SAVE_CALIBRE_PROJECTOR, f"{i}.jpg"))