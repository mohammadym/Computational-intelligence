# Q1.3_graded
# Do not change the above line.

# This cell is for your imports.

!wget "https://www.cufonfonts.com/download/rf/arial"
!pwd
!mkdir -p dataset
!mv arial dataset/
!cd dataset
!unzip dataset/arial -d dataset/
!mv dataset/ARIAL.TTF /usr/share/fonts/truetype/
!cd ..
!rm -rf dataset/

# Q1.3_graded
# Do not change the above line.

# This cell is for your codes.
import os         
import numpy as np
from PIL import Image, ImageFont
import matplotlib.pyplot as plt


def calculate_accuracy(patterns, noise_templates, weights):
  sum = 0
  for index, item in enumerate(noise_templates):
    sign_im = np.sign(np.dot(item, weights))
    sum += np.sum(sign_im == patterns[index]) / patterns[index].shape[0]
  return sum / len(noise_templates)


def noise_template(templates, noise):
  max_pixel = templates[0].shape
  count_noise = int(max_pixel[0] * noise)
  result = []
  for index, template in enumerate(templates):
    for i in np.random.randint(max_pixel[0], size=count_noise):
      templates[index][i] = template[i] * -1
    result.append(template)
  return result


def calculate_weights(templates, lenght):
  weights = np.ndarray(lenght,
                       buffer=np.full(lenght[0] * lenght[1], 0), dtype=float)
  for item in templates: 
    for i in range(lenght[0]):
      for j in range(i):
        weights[i, j] += item[i] * item[j]
        weights[j, i] = weights[i, j] 
  return weights


def create_templates(size):
  font = ImageFont.truetype("/usr/share/fonts/truetype/ARIAL.TTF", size)
  max_pixel = (-1,)
  templates = []
  for item in "ABCDEFGHIJ":
    attribute = font.getmask(item)
    temp_image = Image.Image()._new(attribute)
    data_of_img = temp_image.getdata()
    pixels_of_img = np.array(data_of_img)
    if pixels_of_img.shape[0] > max_pixel[0]:
      max_pixel = pixels_of_img.shape
    templates.append(pixels_of_img)
  final_templates = []
  for item in templates:
    item_size = len(item)
    if item_size == max_pixel[0]:
      continue
    size_of_shift = max_pixel[0] - item_size
    final_template = np.zeros(max_pixel)
    final_template[int(size_of_shift/2):int(len(item) + size_of_shift/2)] = item
    final_template = np.where(final_template > 0, 1, final_template)
    final_templates.append(np.where(final_template == 0, -1, final_template))
  return final_templates

noise1 = 0.1
noise2 = 0.3
noise3 = 0.6


first_font = create_templates(16)
second_font = create_templates(32)
third_font = create_templates(64)


first_weights = calculate_weights(first_font, (first_font[0].shape[0],
                                                 first_font[0].shape[0]))
second_weights = calculate_weights(second_font, (second_font[0].shape[0],
                                                  second_font[0].shape[0]))
third_weights = calculate_weights(third_font, (third_font[0].shape[0],
                                                 third_font[0].shape[0]))


print("first noise = " + str(noise1))
print(calculate_accuracy(third_font,
                         noise_template(third_font, noise1),
                         third_weights))
print(calculate_accuracy(second_font,
                         noise_template(second_font, noise1),
                         second_weights))
print(calculate_accuracy(first_font,
                         convert_to_noisy_pattern(first_font, noise1),
                         first_weights))

print("second noise = " + str(noise2))
print(calculate_accuracy(third_font,
                         noise_template(third_font, noise2),
                         third_weights))
print(calculate_accuracy(second_font,
                         noise_template(second_font, noise2),
                         second_weights))
print(calculate_accuracy(first_font,
                         noise_template(first_font, noise2),
                         first_weights))

print("third noise = " + str(noise3))
print(calculate_accuracy(third_font,
                         noise_template(third_font, noise3), 
                         third_weights))
print(calculate_accuracy(second_font,
                         noise_template(second_font, noise3),
                         second_weights))
print(calculate_accuracy(first_font,
                         noise_template(first_font, noise3),
                         first_weights))


