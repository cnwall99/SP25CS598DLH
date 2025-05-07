import tensorflow as tf
from tensorflow.keras import backend as K

class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, from_logits=False, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.from_logits = from_logits
        
    def call(self, y_true, y_pred):
        if self.from_logits:
            y_pred = tf.nn.sigmoid(y_pred)
            
        bce = K.binary_crossentropy(y_true, y_pred)
        p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
        alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        focal_term = K.pow(1 - p_t, self.gamma)
        loss = alpha_t * focal_term * bce
        
        return K.mean(loss)

class CombinedBCEFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, bce_weight=0.5, focal_weight=0.5, from_logits=False, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.from_logits = from_logits
        
    def call(self, y_true, y_pred):
        if self.from_logits:
            y_pred = tf.nn.sigmoid(y_pred)
            
        bce = K.binary_crossentropy(y_true, y_pred)
        
        p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
        alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        focal_term = K.pow(1 - p_t, self.gamma)
        focal = alpha_t * focal_term * bce
        
        combined_loss = (self.bce_weight * bce) + (self.focal_weight * focal)
        
        return K.mean(combined_loss)