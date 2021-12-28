import pickle
import re
from ml.transformer.vectorizations.config import *
from tensorflow.keras.layers import TextVectorization
import tensorflow as tf



def custom_standardization(input_string):
    punctuation = "!"#$%&'()*+, -./:;<=>?@[\]^`{|}/~"
    strip_chars = punctuation + "¿"
    strip_chars = strip_chars.replace("[", "")
    strip_chars = strip_chars.replace("]", "")
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")

def answer_vector(path):
    from_disk = pickle.load(open(path, "rb"))
    answer_vectorization = TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=sequence_length + 1,
        standardize= custom_standardization,
    )
    answer_vectorization.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
    answer_vectorization.set_weights(from_disk['weights'])

    return answer_vectorization
