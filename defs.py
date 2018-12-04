langs = [
  'java',
  'cpp',
  'python',
  'scala',
  'clojure'
]

file_chars_trunc_limit = 2 * 1024  # 2KB

# Quantization according to LeCun paper (2016): "Text Understanding from Scratch"
supported_chars = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'/\|_@#$%^&*~`+-=<>()[]{}\" " # 70 chars including space
supported_chars_map = {} # pre-calculated vectors
pad_vector = [0 for x in supported_chars]


def _setup():
  i = 0
  for ch in supported_chars:
    vec = [0 for x in supported_chars]
    vec[i] = 1
    supported_chars_map[ch] = vec
    i += 1


_setup()
