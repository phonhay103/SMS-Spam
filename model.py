from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, TimeDistributed, Flatten
from tensorflow.keras.models import Model
from utils import *

def model(stack_fn, max_word, embedding_size, max_len):
    inputs = Input(shape=[max_len])
    x = Embedding(max_word, embedding_size, input_length=max_len)(inputs)
    x = stack_fn(x)
    x = Dense(1, activation='sigmoid')(x)
    return Model(inputs=inputs, outputs=x)

def RNN_Dense(max_word=MAX_WORD, embedding_size=EMBEDDING_SIZE, max_len=MAX_LEN):
    def stack_fn(x):
        x = LSTM(64)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.5)(x)
        return x
    
    return model(stack_fn, max_word, embedding_size, max_len)

def RNN_TimeDis(max_word=MAX_WORD, embedding_size=EMBEDDING_SIZE, max_len=MAX_LEN):
    def stack_fn(x):
        x = LSTM(64, return_sequences=True)(x)
        x = TimeDistributed(Dense(32, activation='relu'))(x)
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        return x
    
    return model(stack_fn, max_word, embedding_size, max_len)