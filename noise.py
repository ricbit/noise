import cv2
import itertools
import numpy as np
import sys
import colour
import functools
import random
import scipy.signal
from hilbertcurve.hilbertcurve import HilbertCurve

msx_rgb = np.array([ 
  [0,0,0],
  [0,0,0],
  [33,200,66],
  [94,220,120],
  [84,85,237],
  [125,118,252],
  [212,82,77],
  [66,235,245],
  [252,85,84],
  [255,121,120],
  [212,193,84],
  [230,206,128],
  [33,176,59],
  [201,91,186],
  [204,204,204],
  [255,255,255]
], dtype=float)

def from3(pattern):
  return ((pattern << 6) + (pattern << 3) + pattern) >> 1

def from2(pattern):
  return pattern * 0b1010101

def median_dim3(image_dim3):
  channels = cv2.split(image_dim3)
  medians = [scipy.signal.medfilt2d(c, kernel_size=3) for c in channels]
  return cv2.merge(medians)

def resize_scr2(image, desired_width, desired_height):
  scale = min(desired_width / image.shape[1],
              desired_height / image.shape[0])
  new_width = int(image.shape[1] * scale)
  new_height = int(image.shape[0] * scale)
  #image = median_dim3(image)
  resized_image = cv2.resize(image, (new_width, new_height))
  #, interpolation=cv2.INTER_NEAREST)
  canvas = np.zeros((desired_height, desired_width, 3), dtype=image.dtype)
  x_offset = (canvas.shape[1] - new_width) // 2
  y_offset = (canvas.shape[0] - new_height) // 2
  canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = (
    resized_image)
  return canvas

def blowup(canvas, scale):
  newdim = (canvas.shape[1] * scale, canvas.shape[0] * scale)
  resized = cv2.resize(canvas, newdim, cv2.INTER_NEAREST)
  return resized

def sRGB_to_Oklab(origin):
  canvas = colour.models.sRGB_to_XYZ(origin.astype(np.float32) / 255.0)
  return colour.models.XYZ_to_Oklab(canvas)

def Oklab_to_sRGB(origin):
  canvas = colour.models.Oklab_to_XYZ(origin)
  return (colour.models.XYZ_to_sRGB(canvas) * 255).astype(np.uint8)

msx_oklab = np.array(sRGB_to_Oklab(np.array(msx_rgb, dtype=float)))

def gen_scr8_palette():
  palette = []
  for r in range(8):
    for g in range(8):
      for b in range(4):
        palette.append((from3(r), from3(g), from2(b)))
  return np.array(sRGB_to_Oklab(np.array(palette, dtype=float)))

def closest_color(color, palette):
  dist = np.linalg.norm(palette - color, axis=1)
  index = np.argmin(dist)
  return palette[index]

@functools.lru_cache(maxsize=None)
def msx_rgb_func(color):
  return closest_color(color, msx_rgb)

@functools.lru_cache(maxsize=None)
def msx_oklab_func(color):
  return closest_color(color, msx_oklab)

@functools.lru_cache(maxsize=None)
def msx_oklab_scr8_func(color):
  return closest_color(color, gen_scr8_palette())

def quantize_2d(image, palette_func):
  (height, width) = image.shape[:2]
  canvas = np.copy(image)
  floyd = np.array([[0.0, 0.0, 7.0], [3.0, 5.0, 1.0]], dtype=float) / 16
  for j in range(height):
    for i in range(width):
      canvas[j][i] = palette_func(tuple(image[j][i]))
      error = image[j][i] - canvas[j][i]
      for dx in [0, 1, 2]:
        for dy in [0, 1]:
          if 0 <= i + dx - 1 < width and 0 <= j + dy < height:
            image[j + dy][i + dx -1] += 0.8 * error * floyd[dy][dx]
  return canvas

def quantize_1d(image, palette_func, curve):
  square = np.zeros((256, 256, 3), dtype=image.dtype)
  square[32:32+192,0:256] = image
  canvas = np.copy(square)
  size = 5
  coefs = np.array([2**i for i in range(size - 1, -1, -1)]) / (2 ** size)
  error = [np.array([0.0,0.0,0.0]) for _ in range(size)]
  for j, i in curve:
    canvas[j][i] = palette_func(tuple(square[j][i] + 0.8 * error[0]))
    pixel_error = square[j][i] - canvas[j][i]
    error[0:size - 1] = error[1:size]
    error[size - 1] = np.array([0,0,0], dtype=float)
    for k in range(size):
      error[k] += coefs[k] * pixel_error
  return canvas

def gen_hilbert():
  bits, dimensions = 8, 2
  curve = HilbertCurve(bits, dimensions)
  size = 2 ** bits
  yield from (curve.point_from_distance(d) for d in range(size * size))

def draw_curve(canvas, curve, scale):
  polyline = np.array(list(curve)).reshape(-1, 1, 2) * 4
  cv2.polylines(canvas, [polyline], isClosed=False, color=(255,0,255),
    thickness=1)
  return canvas

def average_pixel(canvas, x, y, size):
  x0 = max(0, x - size)
  y0 = max(0, y - size)
  x1 = min(255, x + size)
  y1 = min(191, y + size)
  area = (y1 - y0 + 1) * (x1 - x0 + 1)
  return np.median(canvas[y0:y1 + 1, x0:x1 + 1], axis=(0, 1)), area

def quantize_random(image, palette_func):
  canvas = np.zeros((192, 256, 3), dtype=float)
  #canvas = quantize_2d(image, msx_rgb_func)
  for frame in range(10000):
    size = random.randint(0,4)
    area = (2 * size + 1) ** 2
    print(frame)
    for i in range(1000):
      x = random.randint(0, 255)
      y = random.randint(0, 191)
      av, area = average_pixel(canvas, x, y, size)
      #print(av, area)
      av -= canvas[y][x] / area
      goal = (image[y][x] - av / 5)
      color = palette_func(tuple(goal))
      #print(goal, color)
      #print("ca ", canvas[y][x])
      canvas[y][x] = np.copy(color)
      #print("ca ", canvas[y][x])
    #oklab = Oklab_to_sRGB(canvas)
    canvas2 = Oklab_to_sRGB(canvas)
    canvas2 = blowup(canvas2, 4)
    cv2.imshow('Meanwhilex', canvas2.astype(np.uint8))
    cv2.waitKey(1)

def quantize_random_scr2(image, palette):
  canvas = np.zeros((192, 256, 3), dtype=float)
  #canvas = quantize_2d(image.astype(float), msx_rgb_func)
  canvasvram = np.zeros((192, 256), dtype=np.uint8)
  image = image.astype(float)
  for j in range(192):
    for i in range(256):
      dist = np.linalg.norm(palette - canvas[j][i], axis=1)
      canvasvram[j][i] = np.argmin(dist)
  for frame in range(10000000):
    print(frame)
    size = random.randint(0,2)
    for _ in range(100):
      x = random.randint(0,255)#(100, 163)
      y = random.randint(0,191)#(30, 80)
      xx = x & (~7)
      #print(x,y,xx)
      span = 8
      cav = [average_pixel(canvas, xx+i, y, size)[0] for i in range(span)]
      iav = [average_pixel(image, xx+i, y, size)[0] for i in range(span)]
      #print()
      #print([[int(i) for i in g]for g in cav])
      #print([[int(i) for i in g]for g in iav])
      choose = random.randint(0, len(palette) - 1)
      savecanv = np.copy(canvas[y][xx:xx+8])
      savecanvram = canvasvram[y][xx:xx+8]
      ncolors = len(set(savecanvram))
      if ncolors <= 1 or (ncolors == 2 and choose in savecanvram):
        canvasvram[y][x] = choose
        canvas[y][x] = np.copy(palette[choose])
      else:
        changed = savecanvram[x % 8]
        octet = [(choose if c == changed else c) for c in savecanvram]
        canvas[y][xx:xx+8] = np.array([palette[i] for i in octet], dtype=float)
        canvasvram[y][xx:xx+8] = octet
      nav = [average_pixel(canvas, xx+i, y, size)[0] for i in range(span)]
      original = [np.linalg.norm(cav[i] - iav[i]) for i in range(span)]
      proposed = [np.linalg.norm(nav[i] - iav[i]) for i in range(span)]
      if np.linalg.norm(proposed) >= np.linalg.norm(original):
        canvas[y][xx:xx+8] = np.copy(savecanv)
        canvasvram[y][xx:xx+8] = np.copy(savecanvram)
    #canvas2 = Oklab_to_sRGB(canvas)
    canvas2 = blowup(canvas, 4)
    cv2.imshow('Meanwhile', canvas2.astype(np.uint8))
    cv2.waitKey(1)


def quantize_random_oklab(image, palette):
  canvas = np.zeros((192, 256, 3), dtype=float)
  #canvas = quantize_2d(image, msx_oklab_func)
  pixels = list(itertools.product(range(256), range(192)))
  for frame in range(10000):
    print(frame)
    canvas2 = Oklab_to_sRGB(canvas)
    canvas2 = blowup(canvas2, 4)
    cv2.imshow('Meanwhile', canvas2)
    cv2.waitKey(1)
    random.shuffle(pixels)
    for x, y in pixels:
      size = random.randint(0, 0)
      cav, area = average_pixel(canvas, x, y, size)
      iav, area = average_pixel(image, x, y, size)
      choose = random.randint(0, len(palette) - 1)
      save = np.copy(canvas[y][x])
      canvas[y][x] = palette[choose]
      nav, area = average_pixel(canvas, x, y, size)
      canvas[y][x] = np.copy(save)
      #if np.linalg.norm(palette[choose] - image[y][x]) < np.linalg.norm(
      #    canvas[y][x] - image[y][x]):
      if np.linalg.norm(nav - iav) < np.linalg.norm(cav - iav):
        canvas[y][x] = np.copy(palette[choose])

rgb_image = cv2.imread(sys.argv[1])
canvas = rgb_image
#canvas = sRGB_to_Oklab(canvas)
canvas = resize_scr2(canvas, 256, 192)
#canvas1 = resize_scr2(rgb_image, 256, 192)
#canvas1 = quantize(canvas1, msx_rgb_func)
#canvas = quantize_2d(canvas, msx_rgb_func)
#canvas = quantize_2d(canvas, msx_oklab_func)
#canvas = quantize_2d(canvas, msx_oklab_scr8_func)
#canvas = quantize_1d(canvas, msx_oklab_func, gen_hilbert())
#canvas = quantize_1d(canvas, msx_rgb_func, gen_hilbert())
#canvas = quantize_random_oklab(canvas, msx_oklab)
#canvas = quantize_random_oklab(canvas, msx_oklab)
canvas = quantize_random_scr2(canvas, msx_rgb)
#canvas = Oklab_to_sRGB(canvas)
canvas = blowup(canvas, 4)
#canvas = draw_curve(canvas, gen_hilbert(), 4)
#canvas1 = blowup(canvas1, 4)
output = canvas.astype(np.uint8) # np.hstack((canvas1, canvas))

cv2.imshow('Fitted and Padded Image', output)
cv2.waitKey(0)
cv2.destroyAllWindows()
