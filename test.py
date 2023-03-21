import matplotlib.pyplot as plt
from skimage.draw import circle_perimeter_aa
import numpy as np
from PIL import Image
img = np.zeros((220, 220, 3), dtype=np.uint8)
rr, cc, val = circle_perimeter_aa(40, 40, 30)
img[rr, cc] = val

Image.fromarray(img).show()