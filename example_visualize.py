"""
Example program that displays a random dog and cat.

Expects that the files "dog.npy" and "cat.npy" have been downloaded to the
"data/" directory.
"""

import data
import random
import matplotlib.pyplot as plt

dog = data.load_images("dog.npy")
cat = data.load_images("cat.npy")

data.show_image(random.choice(dog))
data.show_image(random.choice(cat))
