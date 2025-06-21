# grad_cam_multimodal.py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# --- Load Model ---
model = tf.keras.models.load_model("checkpoints/orph_multimodal")

# --- Get intermediate model ---
last_conv_layer_name = "conv5_block3_out"  # Update based on CNN encoder
image_input_layer = model.input[3]  # Assuming image input is the 4th input

grad_model = Model(
    inputs=[model.input],
    outputs=[model.get_layer(last_conv_layer_name).output, model.output]
)

def get_img_array(img_path, size):
    img = load_img(img_path, target_size=size)
    array = img_to_array(img)
    return np.expand_dims(array / 255.0, axis=0)

def make_gradcam_heatmap(img_array, model_input):
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(model_input)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlayed = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return overlayed

# --- Run on Sample ---
sample_path = "data/images/sample1.jpg"  # Replace with real image path
img_array = get_img_array(sample_path, (224, 224))

# Mock inputs (token IDs, segments, etc. for 1 sample)
dummy_text_inputs = [
    tf.zeros((1, 256), dtype=tf.int32),  # input_ids
    tf.zeros((1, 256), dtype=tf.int32),  # segment_ids
    tf.zeros((1, 256), dtype=tf.int32),  # modality_ids
    tf.convert_to_tensor(img_array)     # image
]

heatmap = make_gradcam_heatmap(img_array, dummy_text_inputs)
result = overlay_heatmap(sample_path, heatmap)

plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Grad-CAM Visualization")
plt.show()
