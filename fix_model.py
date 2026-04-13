import tensorflow as tf
import os

path = "app/ai_models/weights/ct_stroke_model.h5"

print("1. Loading broken model...")
model = tf.keras.models.load_model(path)

print("2. Fixing the broken math layer...")
# Create a clone of the model but switch the last layer's activation to 'sigmoid'
def fix_activation(layer):
    config = layer.get_config()
    # Check if this is the very last layer
    if layer.name == model.layers[-1].name:
        config['activation'] = 'sigmoid'
    return layer.__class__.from_config(config)

fixed_model = tf.keras.models.clone_model(model, clone_function=fix_activation)

# Transfer all the trained knowledge to the fixed model
fixed_model.set_weights(model.get_weights())

print("3. Saving the fixed model...")
fixed_model.save(path)
print("Surgery complete! Your CT model is now fixed.")