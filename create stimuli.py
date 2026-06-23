# create stimuli

from PIL import Image
import os

os.makedirs('practice_faces', exist_ok=True)
os.makedirs('experimental_faces', exist_ok=True)

for i in range(20):
    img = Image.new('RGB', (200, 250), color=(100+i*5, 100, 150))
    img.save(f'practice_faces/face_{i}.png')

for i in range(30):
    img = Image.new('RGB', (200, 250), color=(150, 100+i*5, 100))
    img.save(f'experimental_faces/face_{i}.png')