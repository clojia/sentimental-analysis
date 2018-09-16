The project is to automatically identify if customers happy or not from their reviews.
Prioritize those "positive" reviews

There are two methods in industry:  statistics or machine learning. The former one combines different sentimental weights of tokens to get one final score, which means we should have a comprehensive dictionary along with reasonable token weights (one time consuming task indeed)… The method here is using BiGRU+Attention, datasets came from Kaggle: Hotel Reviews.
### Steps:

##### Text preprocessing [word2vec]
```
texts = []
labels = []
for idx in range(data_train.review.shape[0]):
    text = BeautifulSoup(data_train.review[idx], "lxml")
    texts.append(clean_str(text.get_text().encode('ascii','ignore')))
    labels.append(data_train.sentiment[idx])
GLOVE_DIR = "/data/mpk"
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Total %s word vectors.' % len(embeddings_index))
labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', len(texts))
print('Shape of label tensor:', len(labels))
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
```

##### Split training/test sets:
```
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]
```
##### Embedding layer:
```
embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
print ('Length of embedding_matrix:', embedding_matrix.shape[0])
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            mask_zero=False,
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
print('Traing and validation set number of positive and negative reviews')
print y_train.sum(axis=0)
print y_val.sum(axis=0)
BiGRU + Attention + dense layers:

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
l_gru = Bidirectional(LSTM(100, return_sequences=True))(embedded_sequences)
l_att = Attention_layer()(l_gru)
dense_1 = Dense(100,activation='tanh')(l_att)
dense_2 = Dense(2, activation='softmax')(dense_1)
```

### Results:
##### Network Structure:
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 1000)              0
_________________________________________________________________
embedding_1 (Embedding)      (None, 1000, 100)         4801900
_________________________________________________________________
bidirectional_1 (Bidirection (None, 1000, 200)         160800
_________________________________________________________________
attention_layer_1 (Attention (None, 200)               40200
_________________________________________________________________
dense_1 (Dense)              (None, 100)               20100
_________________________________________________________________
dense_2 (Dense)              (None, 2)                 202
=================================================================
```

##### Training Output:
```
Train on 31146 samples, validate on 7786 samples
Epoch 1/10
31146/31146 [==============================] - 2255s 72ms/step - loss: 0.4864 - acc: 0.7715 - val_loss: 0.3460 - val_acc: 0.8495
Epoch 2/10
31146/31146 [==============================] - 2288s 73ms/step - loss: 0.3370 - acc: 0.8580 - val_loss: 0.3055 - val_acc: 0.8726
Epoch 3/10
31146/31146 [==============================] - 2388s 77ms/step - loss: 0.3025 - acc: 0.8748 - val_loss: 0.3026 - val_acc: 0.8713
Epoch 4/10
31146/31146 [==============================] - 1930s 62ms/step - loss: 0.2827 - acc: 0.8807 - val_loss: 0.2785 - val_acc: 0.8854
Epoch 5/10
31146/31146 [==============================] - 2125s 68ms/step - loss: 0.2659 - acc: 0.8905 - val_loss: 0.2853 - val_acc: 0.8827
Epoch 6/10
31146/31146 [==============================] - 2131s 68ms/step - loss: 0.2501 - acc: 0.8977 - val_loss: 0.2706 - val_acc: 0.8899
Epoch 7/10
31146/31146 [==============================] - 2209s 71ms/step - loss: 0.2354 - acc: 0.9057 - val_loss: 0.2740 - val_acc: 0.8892
Epoch 8/10
31146/31146 [==============================] - 2414s 78ms/step - loss: 0.2172 - acc: 0.9124 - val_loss: 0.2666 - val_acc: 0.8919
Epoch 9/10
31146/31146 [==============================] - 2561s 82ms/step - loss: 0.2008 - acc: 0.9207 - val_loss: 0.3291 - val_acc: 0.8813
Epoch 10/10
31146/31146 [==============================] - 2436s 78ms/step - loss: 0.1826 - acc: 0.9297 - val_loss: 0.3125 - val_acc: 0.8835
```
