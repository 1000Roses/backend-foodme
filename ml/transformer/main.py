
from ml.transformer.transformer.main import transformer
from ml.transformer.vectorizations.answer_vector import answer_vector
from ml.transformer.vectorizations.question_vector import question_vector
from ml.w2seg.main import w2seg
import numpy as np


answer_vectorization = answer_vector(path = "ml/transformer/assets/answer_vectorization.pkl")
question_vectorization = question_vector(path = "ml/transformer/assets/question_vectorization.pkl")
transformer.load_weights("ml/transformer/assets/transformer_80epochs_weights.h5")

answer_vocab = answer_vectorization.get_vocabulary()
answer_index_lookup = dict(zip(range(len(answer_vocab)), answer_vocab))
max_decoded_sentence_length = 20

def inference(input_sentence):
    # make word segment here
    # input_sentence = w2seg(input_sentence)
    #
    tokenized_input_sentence = question_vectorization([input_sentence])
    decoded_sentence = "[start]"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = answer_vectorization([decoded_sentence])[:, :-1]
        predictions = transformer([tokenized_input_sentence, tokenized_target_sentence])

        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = answer_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token

        if sampled_token == "[end]":
            break
    return decoded_sentence

