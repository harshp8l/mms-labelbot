import os
import codecs
import defs
import re
import numpy as np
import requests

def text_to_vec(text, file_vector_size=10 * 1024, normalise_whitespace=True):
  """
  extracts feature vector from text
  :param text: text
  :param file_vector_size: size of the vector
  :param normalise_whitespace: replacing all whitespace to space
  :return: vector
  """
  file_vector = []  # will be byte array
  # Normalizing whitespace
  # Tradeoff: This can be fatal for whitespace significant languages however this gives access to more code data
  if normalise_whitespace:
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = re.sub('\s+', ' ', text)

  text = text[0:file_vector_size]
  for ch in text:
    if ch in defs.supported_chars_map:
      file_vector.append(defs.supported_chars_map[ch])

  if len(file_vector) < file_vector_size:
    for j in range(0, file_vector_size - len(file_vector)):
      file_vector.append(defs.pad_vector)

  return np.array(file_vector)


def get_snippets(file_name):
  """
  :param file_name: name of the file
  :return: snippet data of the file
  """
  with codecs.open(file_name, mode='r', encoding='utf-8') as f:
    text = f.read().lower()
  return [text]


def file_to_vec(file_name, file_vector_size=10 * 1024, normalise_whitespace=True):
  texts = get_snippets(file_name)
  return [text_to_vec(t, file_vector_size, normalise_whitespace) for t in texts]

