import tensorflow as tf

AUG_BRIGHTNESS_RANGE = (0.85, 1.15)
AUG_CONTRAST_RANGE = (0.85, 1.15)
AUG_SATURATION_RANGE = (0.85, 1.15)
AUG_HUE_MAX_DELTA = 0.02


def augment_training_image(image, clip_min=0.0, clip_max=1.0):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    image = tf.image.rot90(image, k=k)

    brightness_factor = tf.random.uniform([], AUG_BRIGHTNESS_RANGE[0], AUG_BRIGHTNESS_RANGE[1])
    image = image * brightness_factor

    image = tf.image.random_contrast(image, AUG_CONTRAST_RANGE[0], AUG_CONTRAST_RANGE[1])
    image = tf.image.random_saturation(image, AUG_SATURATION_RANGE[0], AUG_SATURATION_RANGE[1])
    image = tf.image.random_hue(image, AUG_HUE_MAX_DELTA)

    image = tf.clip_by_value(image, clip_min, clip_max)
    return image