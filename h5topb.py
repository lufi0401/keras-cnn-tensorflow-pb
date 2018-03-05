#!/usr/bin/env python
# -.- coding: utf-8 -.-
import os
import keras
import keras.backend as K
import tensorflow as tf



def reset_keras():
    # reset session
    K.clear_session()
    sess = tf.Session()
    K.set_session(sess)

def prepare_keras():
    # disable loading of learning nodes
    K.set_learning_phase(0)

def keras_applications_preprocess_fn(img):
    img = img[..., ::-1]
    mean = tf.constant([[[103.939, 116.779, 123.68]]])
    img -= mean
    return img

def prepare_model(base_model, img_size=(224, 224), class_labels=None, top_k=None):

    key_placeholder = tf.placeholder_with_default(['0'], shape=(None, ), name='key')
    # intake images and preprocess them
    img_bytes = tf.placeholder(tf.string, shape=(None,), name='img_bytes')
    img_loaded = tf.map_fn(lambda x: tf.image.decode_jpeg(x, channels=3), img_bytes, dtype=tf.uint8)
    img = tf.image.resize_images(img_loaded, img_size)
    # img_std = tf.map_fn(tf.keras.applications.resnet50.preprocess_input, img)
    img_std = tf.map_fn(keras_applications_preprocess_fn, img)
    img_input = tf.reshape(img_std, (-1,) + img_size + (3,))
    # bridging input and model
    base_model_output = base_model(img_input)
    key_out = tf.identity(key_placeholder, name='key_out')
    # define outputs
    if class_labels is None:
        outputs = {'key': key_out,
                   tf.saved_model.signature_constants.PREDICT_OUTPUTS:
                        base_model_output}
    else:
        if top_k is None:
            top_k = len(class_labels)
        elif top_k > len(class_labels):
            top_k = len(class_labels)
        labels = tf.constant(class_labels, dtype=tf.string)
        output_values, indices = tf.nn.top_k(base_model_output, k=top_k, sorted=True)
        output_labels = tf.gather(labels, indices, axis=0)
        outputs = {'key': key_out,
                   tf.saved_model.signature_constants.PREDICT_OUTPUTS:
                        output_labels,
                   'confidence': output_values}
    # define inputs
    inputs = {'key': key_placeholder,
              tf.saved_model.signature_constants.PREDICT_INPUTS: img_bytes}
    return inputs, outputs



def export_models(path, inputs, outputs):
    # export saved model
    # export_path = '../trained_models/gender_models' + '/export/' + str(random.randint(0,99999999))
    builder = tf.saved_model.builder.SavedModelBuilder(path)

    signature = tf.saved_model.signature_def_utils.predict_signature_def(inputs=inputs,
                                      outputs=outputs)

    with K.get_session() as sess:
        builder.add_meta_graph_and_variables(
            sess=sess,
            tags=[tf.saved_model.tag_constants.SERVING],
            signature_def_map={
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature
            }
        )
        builder.save()


if __name__ == '__main__':
    # Test script
    model_path = '/dir/to/keras/model'
    export_path = '/dir/to/save/pb'
    prepare_keras()
    base_model = keras.models.load_model(model_path, compile=False)
    inputs, outputs = prepare_model(base_model)
    export_models(export_path, inputs, outputs)
