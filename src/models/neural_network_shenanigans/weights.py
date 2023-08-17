''' Get the weights of a neural network'''

from pathlib import Path
import tensorflow as tf

def get_weights(model):
    ''' Get the weights of a neural network'''
    weights = []
    for layer in model.layers:
        layer_weights = layer.get_weights()
        if layer_weights:
            weights.append(layer_weights)
    return weights

# model_name = '20230807-120711_0.863 (BEST).h5'
model_name = '20230808-123801_0.846 (BEST2).h5'
model_path = Path(__file__).resolve(
).parent / 'csv' / 'savedModels' / 'neural_network_better_split_new' / model_name

tf_model = tf.keras.models.load_model(model_path)

# tf_model.summary()

model_weights = get_weights(tf_model)

for layer in model_weights[0][0]:
    print(f'{layer[2]}, {layer[3]}, {layer[4]}')
