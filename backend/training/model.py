import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.applications.vgg16 import VGG16

@tf.keras.utils.register_keras_serializable(package="MSGF")
class ScaleGatedFeatureFusionUnit(layers.Layer):
    """
    Scale-Gated Feature Fusion Unit (SGFU)
    
    Dynamically fuses multi-scale feature maps (F3, F4, F5) by learning
    adaptive scale importance weights [alpha3, alpha4, alpha5] conditioned on
    both average and peak spatial activations (GAP + GMP).
    """
    def __init__(self, mlp_hidden1=128, mlp_hidden2=64, l2_reg=1e-4, **kwargs):
        super(ScaleGatedFeatureFusionUnit, self).__init__(**kwargs)
        self.mlp_hidden1 = mlp_hidden1
        self.mlp_hidden2 = mlp_hidden2
        self.l2_reg = l2_reg
        
        # Dense layers for gating MLP
        self.dense1 = layers.Dense(
            mlp_hidden1,
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg) if l2_reg else None,
            name=f"{self.name}_dense1"
        )
        self.dense2 = layers.Dense(
            mlp_hidden2,
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg) if l2_reg else None,
            name=f"{self.name}_dense2"
        )
        self.logits = layers.Dense(
            3,
            activation=None,
            name=f"{self.name}_logits"
        )
        self.softmax = layers.Softmax(name=f"{self.name}_softmax")

    def call(self, inputs):
        """
        inputs: list of 3 tensors [F3, F4, F5], each of shape (batch, 16, 16, 128)
        returns: fused tensor F_fused of shape (batch, 16, 16, 128)
        """
        F3, F4, F5 = inputs
        
        # Step 2: GAP and GMP for each feature map
        gap3 = tf.reduce_mean(F3, axis=[1, 2])  # (batch, 128)
        gmp3 = tf.reduce_max(F3, axis=[1, 2])   # (batch, 128)
        
        gap4 = tf.reduce_mean(F4, axis=[1, 2])  # (batch, 128)
        gmp4 = tf.reduce_max(F4, axis=[1, 2])   # (batch, 128)
        
        gap5 = tf.reduce_mean(F5, axis=[1, 2])  # (batch, 128)
        gmp5 = tf.reduce_max(F5, axis=[1, 2])   # (batch, 128)
        
        # Step 3: Concatenate pooled descriptors -> (batch, 768)
        descriptors = tf.concat([gap3, gmp3, gap4, gmp4, gap5, gmp5], axis=-1)
        
        # Step 4: Gating MLP
        x = self.dense1(descriptors)
        x = self.dense2(x)
        
        # Step 5: Generate three scale logits [g3, g4, g5]
        scale_logits = self.logits(x)
        
        # Step 6: Softmax to obtain normalized scale weights [alpha3, alpha4, alpha5]
        weights = self.softmax(scale_logits)  # (batch, 3)
        
        # Reshape weights for broadcasting with spatial dimensions (batch, 1, 1, 1)
        alpha3 = tf.reshape(weights[:, 0], (-1, 1, 1, 1))
        alpha4 = tf.reshape(weights[:, 1], (-1, 1, 1, 1))
        alpha5 = tf.reshape(weights[:, 2], (-1, 1, 1, 1))
        
        # Step 7: Weighted feature fusion: F_fused = alpha3*F3 + alpha4*F4 + alpha5*F5
        f_fused = alpha3 * F3 + alpha4 * F4 + alpha5 * F5
        return f_fused

    def get_config(self):
        config = super(ScaleGatedFeatureFusionUnit, self).get_config()
        config.update({
            "mlp_hidden1": self.mlp_hidden1,
            "mlp_hidden2": self.mlp_hidden2,
            "l2_reg": self.l2_reg
        })
        return config


def build_ms_gf_vgg16(input_shape=(128, 128, 3), num_classes=4, projection_filters=128, l2_reg=1e-4):
    """
    Builds the complete MS-GF-VGG16 architecture.
    
    1. VGG16 Backbone with ImageNet weights
    2. Multi-scale feature extraction: Block 3, Block 4, Block 5
    3. 1x1 Convolution projection to 128 channels
    4. Spatial alignment to 16x16
    5. Scale-Gated Feature Fusion Unit (SGFU)
    6. Classification Head
    """
    inputs = layers.Input(shape=input_shape, name="input_image")
    
    # 3.1 VGG16 Backbone (ImageNet weights)
    backbone = VGG16(include_top=False, weights="imagenet", input_tensor=inputs)
    
    # Freeze the initial 10 layers
    for layer in backbone.layers[:10]:
        layer.trainable = False
        
    # Extract feature maps:
    # Block 3: 16 x 16 x 256
    # Block 4: 8 x 8 x 512
    # Block 5: 4 x 4 x 512
    b3 = backbone.get_layer("block3_pool").output
    b4 = backbone.get_layer("block4_pool").output
    b5 = backbone.get_layer("block5_pool").output
    
    # 3.2 1x1 Convolutional Projections to 128 channels
    reg = regularizers.l2(l2_reg) if l2_reg else None
    
    f3 = layers.Conv2D(projection_filters, (1, 1), padding='same', activation='relu',
                       kernel_regularizer=reg, name="proj_block3")(b3)
    f4 = layers.Conv2D(projection_filters, (1, 1), padding='same', activation='relu',
                       kernel_regularizer=reg, name="proj_block4")(b4)
    f5 = layers.Conv2D(projection_filters, (1, 1), padding='same', activation='relu',
                       kernel_regularizer=reg, name="proj_block5")(b5)
    
    # Spatial Alignment to 16 x 16
    # F3: Keep 16 x 16
    # F4: Upsample x2 -> 16 x 16
    # F5: Upsample x4 -> 16 x 16
    f4_aligned = layers.UpSampling2D(size=(2, 2), interpolation='bilinear', name="align_block4")(f4)
    f5_aligned = layers.UpSampling2D(size=(4, 4), interpolation='bilinear', name="align_block5")(f5)
    
    # 3.3 Scale-Gated Feature Fusion Unit (SGFU)
    sgfu = ScaleGatedFeatureFusionUnit(mlp_hidden1=128, mlp_hidden2=64, l2_reg=l2_reg, name="sgfu")
    f_fused = sgfu([f3, f4_aligned, f5_aligned])
    
    # 3.4 Classification Head
    x = layers.GlobalAveragePooling2D(name="fused_gap")(f_fused)
    x = layers.Dense(256, activation='relu', kernel_regularizer=reg, name="head_dense1")(x)
    x = layers.Dropout(0.5, name="head_dropout1")(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=reg, name="head_dense2")(x)
    x = layers.Dropout(0.3, name="head_dropout2")(x)
    outputs = layers.Dense(num_classes, activation='softmax', name="predictions")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="MS-GF-VGG16")
    return model


def build_standard_vgg16(input_shape=(128, 128, 3), num_classes=4, l2_reg=1e-4):
    """Ablation Experiment 1: Standard pretrained VGG16 with only block5 features."""
    inputs = layers.Input(shape=input_shape, name="input_image")
    backbone = VGG16(include_top=False, weights="imagenet", input_tensor=inputs)
    for layer in backbone.layers[:10]:
        layer.trainable = False
        
    b5 = backbone.get_layer("block5_pool").output
    reg = regularizers.l2(l2_reg) if l2_reg else None
    
    x = layers.GlobalAveragePooling2D(name="vgg_gap")(b5)
    x = layers.Dense(256, activation='relu', kernel_regularizer=reg, name="head_dense1")(x)
    x = layers.Dropout(0.5, name="head_dropout1")(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=reg, name="head_dense2")(x)
    x = layers.Dropout(0.3, name="head_dropout2")(x)
    outputs = layers.Dense(num_classes, activation='softmax', name="predictions")(x)
    
    return Model(inputs=inputs, outputs=outputs, name="Standard-VGG16")


def build_multiscale_vgg16_no_gating(input_shape=(128, 128, 3), num_classes=4, projection_filters=128, l2_reg=1e-4):
    """Ablation Experiment 2: VGG16 with Multi-Scale Feature Fusion without adaptive gating (static average)."""
    inputs = layers.Input(shape=input_shape, name="input_image")
    backbone = VGG16(include_top=False, weights="imagenet", input_tensor=inputs)
    for layer in backbone.layers[:10]:
        layer.trainable = False
        
    b3 = backbone.get_layer("block3_pool").output
    b4 = backbone.get_layer("block4_pool").output
    b5 = backbone.get_layer("block5_pool").output
    
    reg = regularizers.l2(l2_reg) if l2_reg else None
    f3 = layers.Conv2D(projection_filters, (1, 1), padding='same', activation='relu', kernel_regularizer=reg)(b3)
    f4 = layers.Conv2D(projection_filters, (1, 1), padding='same', activation='relu', kernel_regularizer=reg)(b4)
    f5 = layers.Conv2D(projection_filters, (1, 1), padding='same', activation='relu', kernel_regularizer=reg)(b5)
    
    f4_aligned = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(f4)
    f5_aligned = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(f5)
    
    # Static unweighted feature fusion (simple average)
    f_fused = layers.Average()([f3, f4_aligned, f5_aligned])
    
    x = layers.GlobalAveragePooling2D(name="fused_gap")(f_fused)
    x = layers.Dense(256, activation='relu', kernel_regularizer=reg, name="head_dense1")(x)
    x = layers.Dropout(0.5, name="head_dropout1")(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=reg, name="head_dense2")(x)
    x = layers.Dropout(0.3, name="head_dropout2")(x)
    outputs = layers.Dense(num_classes, activation='softmax', name="predictions")(x)
    
    return Model(inputs=inputs, outputs=outputs, name="MS-VGG16-NoGating")

