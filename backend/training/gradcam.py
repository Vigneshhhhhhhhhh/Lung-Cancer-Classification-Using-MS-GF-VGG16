import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.cm as cm


def compute_gradcam(model, img_array, target_layer_name=None, pred_index=None):
    """
    Computes Grad-CAM heatmap for a given input image and trained MS-GF-VGG16 model.
    
    Args:
        model: Trained tf.keras.Model
        img_array: Preprocessed image tensor of shape (1, H, W, 3)
        target_layer_name: Name of the convolutional or feature layer. If None, automatically selects
                           'sgfu' or 'block5_conv3'.
        pred_index: Class index to compute gradients for. If None, uses top predicted class.
        
    Returns:
        heatmap: 2D numpy array of values in [0, 1]
    """
    if target_layer_name is None:
        # Default to 'sgfu' or 'block5_conv3' if available
        available_layers = [layer.name for layer in model.layers]
        if "sgfu" in available_layers:
            target_layer_name = "sgfu"
        elif "block5_conv3" in available_layers:
            target_layer_name = "block5_conv3"
        elif "proj_block5" in available_layers:
            target_layer_name = "proj_block5"
        else:
            # Fallback to last Conv2D layer
            conv_layers = [l.name for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)]
            target_layer_name = conv_layers[-1] if conv_layers else model.layers[-4].name

    target_layer = model.get_layer(target_layer_name)

    # Build a gradient model that maps the inputs to the activations of target_layer and the model predictions
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[target_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Compute gradient of the predicted class with respect to the output feature map of target_layer
    grads = tape.gradient(class_channel, conv_outputs)

    # Global average pooling of gradients: mean intensity of gradient per channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the channels of the feature map by the pooled gradients
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU to retain only features that have a positive influence on the class
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon())
    return heatmap.numpy()


def overlay_gradcam(original_img_pil: Image.Image, heatmap: np.ndarray, alpha=0.4, colormap="jet"):
    """
    Overlays a Grad-CAM heatmap onto the original PIL image.
    
    Args:
        original_img_pil: PIL Image instance (RGB)
        heatmap: 2D numpy array of values in [0, 1]
        alpha: Opacity of heatmap overlay (0.0 to 1.0)
        colormap: Matplotlib colormap name (e.g. 'jet', 'viridis', 'plasma')
        
    Returns:
        overlay_pil: PIL Image with superimposed heatmap
        heatmap_pil: PIL Image of the heatmap alone
    """
    # Resize heatmap to match image dimensions
    w, h = original_img_pil.size
    heatmap_resized = Image.fromarray(np.uint8(255 * heatmap)).resize((w, h), Image.Resampling.BILINEAR)
    heatmap_resized_arr = np.array(heatmap_resized) / 255.0

    # Colorize heatmap
    cmap = cm.get_cmap(colormap)
    colored_heatmap = cmap(heatmap_resized_arr)  # RGBA in [0, 1]
    colored_heatmap = np.uint8(255 * colored_heatmap[..., :3])  # RGB in [0, 255]

    heatmap_pil = Image.fromarray(colored_heatmap)

    # Convert original to RGB array
    orig_arr = np.array(original_img_pil.convert("RGB"))

    # Alpha blend: (1 - alpha) * original + alpha * heatmap
    blended = np.uint8((1 - alpha) * orig_arr + alpha * colored_heatmap)
    overlay_pil = Image.fromarray(blended)

    return overlay_pil, heatmap_pil

