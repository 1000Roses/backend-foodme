import pickle
from ml.transformer.vectorizations.config import *
from tensorflow.keras.layers import TextVectorization
import tensorflow as tf


### open question 
def question_vector(path):

    from_disk = pickle.load(open(path, "rb"))

    question_vectorization = TextVectorization(
        max_tokens=vocab_size, output_mode="int", output_sequence_length=sequence_length,
    )
    question_vectorization.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
    question_vectorization.set_weights(from_disk['weights'])

    return question_vectorization
    