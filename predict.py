import data_transformer
import keras
import sys
import numpy as np
import defs
import mxnet as mx

print('usage: predict.py <filename>')
if len(sys.argv) < 2:
  exit(0)

file_name = sys.argv[1]

x = mx.nd.array(data_transformer.file_to_vec(file_name, file_vector_size=defs.file_chars_trunc_limit))
sym, arg_params, aux_params = mx.model.load_checkpoint(prefix='./prog', epoch=0)
mod = mx.mod.Module(symbol=sym, 
                    data_names=['/dropout_1_input1'], 
                    context=mx.cpu(), 
                    label_names=None)
mod.bind(for_training=False,
         data_shapes=[('/dropout_1_input1',(1, 2048, 70),'float32','NTC')],
         label_shapes=mod._label_shapes)

mod.set_params(arg_params, aux_params)
print(mod.data_names)
print(mod.data_shapes)
print(mod.output_names)
print(mod.output_shapes)
y = mod.predict(x)

for i in range(0, len(defs.langs)):
    print("{} - {}:     {}%".format(' ' if (y[0][i] < 0.5) else '*', defs.langs[i], (100 * y[0][i])).strip('<NDArray 1 @cpu(0)>%'))
