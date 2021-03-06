# -*- coding: utf-8 -*-
"""project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18c5yWMEqLRFB1QLY9gWUt2dBl9O2sNP2
"""

from google.colab import drive
drive.mount('/content/drive')

import re
def clean_eng(text):

    text = text.lower()
    text = text.strip()
    
    text = re.sub(r'[-()]', "", text)
    text = re.sub(r'[.!?,"]', " \1", text)
    
    return text

def clean_chn(text):

    text = re.sub(r"[^\u2E80-\uFE4F\uFF00-\uFFEF]", "", text)

    return text

eng_lines = []
chn_lines_uncut = []
chn_lines_cut = []

with open('/content/drive/My Drive/news-commentary-v13.zh-en.zh') as f:
  for line in f.readlines():
    chn_lines_uncut.append(clean_chn(line))

import jieba
chn_lines_cut = [list(jieba.cut(line.strip())) for line in chn_lines_uncut]
chn_lines_uncut = [list(line) for line in chn_lines_uncut]


with open('/content/drive/My Drive/news-commentary-v13.zh-en.en') as f:
  for line in f.readlines():
    eng_lines.append(clean_eng(line).split())

max_len = 30
selected = []
for i in range(len(eng_lines)):
  if len(eng_lines[i]) <=  max_len and\
     len(chn_lines_uncut[i]) <=  max_len and\
     len(chn_lines_cut[i]) <=  max_len:
    selected.append(i)

eng_lines = [eng_lines[i] for i in selected]
chn_lines_cut = [chn_lines_cut[i] for i in selected]
chn_lines_uncut = [chn_lines_uncut[i] for i in selected]

print('Total:',len(selected))

ch_to_rad = {}
with open('/content/drive/My Drive/chaizi-jt.txt') as f:
  for line in f.readlines():
    line = line.split('\t')
    ch_to_rad[line[0]]=line[1].split()

chn_lines_radical = []
for line in chn_lines_uncut:
  sentence = []
  for ch in line:
    if ch in ch_to_rad.keys():
      sentence+=ch_to_rad[ch]
    else:
      sentence+=[ch]
  chn_lines_radical.append(sentence)

def build_vocab(lines):
  vocab = {}
  for line in lines:
    for word in line:
        if word not in vocab:
            vocab[word] = 1
        else:
            vocab[word] += 1
  return vocab

eng_vocab = build_vocab(eng_lines)
chn_vocab_word = build_vocab(chn_lines_cut)
chn_vocab_char = build_vocab(chn_lines_uncut)

GO_INDEX = 3

codes = ['<PAD>','<EOS>','<UNK>','<GO>']
eng_vocab_size = 4000 + 4 
eng_int_to_word = codes + sorted(eng_vocab.keys(), key = lambda k:-eng_vocab[k])[:eng_vocab_size - 4]
eng_word_to_int = dict(zip(eng_int_to_word,range(eng_vocab_size)))

chn_vocab = chn_vocab_char
chn_lines = chn_lines_uncut
chn_vocab_size = 2000 + 4 
chn_int_to_word = codes + sorted(chn_vocab.keys(), key = lambda k:-chn_vocab[k])[:chn_vocab_size - 4]
chn_word_to_int = dict(zip(chn_int_to_word,range(chn_vocab_size)))

def build_index(lines,int_to_word,word_to_int):
  index = []
  for line in lines:
    sentence = []
    for word in line:
        if word not in int_to_word:
            sentence.append(word_to_int['<UNK>'])
        else:
            sentence.append(word_to_int[word])
    index.append(sentence)
  return index

def select_language(s):
  if s == 'chn':
    return chn_lines,chn_int_to_word,chn_word_to_int
  else:
    return eng_lines,eng_int_to_word,eng_word_to_int

source_lines,src_int_to_word,src_word_to_int = select_language('chn')
source_lines = [['<GO>']+line+['<EOS>'] for line in source_lines]
source_index = build_index(source_lines,src_int_to_word,src_word_to_int)

target_lines,tar_int_to_word,tar_word_to_int = select_language('eng')
target_lines = [['<GO>']+line+['<EOS>'] for line in target_lines]
target_index = build_index(target_lines,tar_int_to_word,tar_word_to_int)

from sklearn.model_selection import train_test_split
source_index_train, source_index_test, target_index_train, target_index_test = train_test_split(source_index, target_index, test_size=0.2)

print('Training:',len(source_index_train))
print('Test:',len(source_index_test))

ENC_VOCAB_SIZE = len(src_int_to_word)
DEC_VOCAB_SIZE = len(tar_int_to_word)
ENC_EMBEDDING_DIM = 256
DEC_EMBEDDING_DIM = 256
UNITS = 1024
MAX_LENGTH = max_len + 2
BUFFER_SIZE = len(source_index)
BATCH_SIZE = 64

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x
import tensorflow as tf
input_tensor = tf.keras.preprocessing.sequence.pad_sequences(source_index_train,maxlen=MAX_LENGTH,padding='post')
target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_index_train,maxlen=MAX_LENGTH,padding='post')
dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

example_input_batch, example_target_batch = next(iter(dataset))
example_input_batch.shape, example_target_batch.shape

encoder_embedding = tf.keras.layers.Embedding(ENC_VOCAB_SIZE, ENC_EMBEDDING_DIM)
encoder_gru = tf.keras.layers.GRU(UNITS,return_sequences=True,return_state=True,recurrent_initializer='glorot_uniform')

encoder_hidden = tf.keras.Input(shape=(UNITS,))
encoder_input = tf.keras.Input(shape=(MAX_LENGTH,))

encoder_embed = encoder_embedding(encoder_input)
encoder_output, encoder_state = encoder_gru(encoder_embed, initial_state = encoder_hidden)

encoder = tf.keras.Model(inputs=[encoder_input,encoder_hidden],outputs=[encoder_output, encoder_state])
encoder.summary()

sample_hidden = tf.zeros((BATCH_SIZE, UNITS))
sample_output, sample_hidden = encoder((example_input_batch, sample_hidden))
print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
print ('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))

decoder_embedding = tf.keras.layers.Embedding(DEC_VOCAB_SIZE, DEC_EMBEDDING_DIM)
decoder_gru = tf.keras.layers.GRU(UNITS,return_sequences=True,return_state=True,recurrent_initializer='glorot_uniform')
decoder_fc = tf.keras.layers.Dense(DEC_VOCAB_SIZE)
decoder_attention = tf.keras.layers.AdditiveAttention()

decoder_input = tf.keras.Input(shape=(1,))
decoder_enchidden = tf.keras.Input(shape=(UNITS,))
decoder_encoutput = tf.keras.Input(shape=(MAX_LENGTH,UNITS))

decoder_context = decoder_attention([tf.expand_dims(decoder_enchidden, 1), decoder_encoutput])
decoder_embed = decoder_embedding(decoder_input)
decoder_concat = tf.concat([decoder_context, decoder_embed], axis=-1)
decoder_output, decoder_state = decoder_gru(decoder_concat)
decoder_reshaped_output = tf.reshape(decoder_output, (-1, decoder_output.shape[2]))
decoder_fc_ouput = decoder_fc(decoder_reshaped_output)

decoder = tf.keras.Model(inputs=[decoder_input,decoder_enchidden,decoder_encoutput],outputs=[decoder_fc_ouput, decoder_state])
decoder.summary()

sample_decoder_output, _= decoder((tf.random.uniform((BATCH_SIZE, 1)), sample_hidden, sample_output))
print ('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)

@tf.function
def train_step(inp, targ, enc_hidden):
  loss = 0
  with tf.GradientTape() as tape:
    enc_output, enc_hidden = encoder((inp, enc_hidden))
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([GO_INDEX] * BATCH_SIZE, 1)
    for t in range(1, targ.shape[1]):
      predictions, dec_hidden = decoder((dec_input, dec_hidden, enc_output))
      loss += loss_function(targ[:, t], predictions)
      dec_input = tf.expand_dims(targ[:, t], 1)
  batch_loss = (loss / int(targ.shape[1]))
  variables = encoder.trainable_variables + decoder.trainable_variables
  gradients = tape.gradient(loss, variables)
  optimizer.apply_gradients(zip(gradients, variables))
  return batch_loss

checkpoint_dir = '/content/training_checkpoints'
checkpoint_prefix = checkpoint_dir+'/ckpt'
checkpoint = tf.train.Checkpoint(optimizer=optimizer,encoder=encoder,decoder=decoder)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

import time,tqdm
EPOCHS = 10
loss_record = []
steps_per_epoch=BUFFER_SIZE//BATCH_SIZE
for epoch in range(EPOCHS):
  start = time.time()
  enc_hidden = tf.zeros((BATCH_SIZE, UNITS))
  total_loss = 0
  epoch_dataset = dataset.take(steps_per_epoch)
  for batch in tqdm.trange(steps_per_epoch):
    inp, targ = next(iter(epoch_dataset))
    batch_loss = train_step(inp, targ, enc_hidden)
    loss_record.append(batch_loss)
    total_loss += batch_loss
    if batch % 100 == 0:
      print('\nEpoch {} Batch {} Loss {:.4f}'.format(epoch + 1,batch,batch_loss.numpy()))     
  
  checkpoint.save(file_prefix = checkpoint_prefix)
  print('Epoch {} Loss {:.4f}'.format(epoch + 1,total_loss / steps_per_epoch))
  print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))



def translate(inputs):

  inputs = tf.convert_to_tensor(inputs)

  result = []

  hidden = [tf.zeros((1, UNITS))]
  enc_out, enc_hidden = encoder(inputs, hidden)

  dec_hidden = enc_hidden
  dec_input = tf.expand_dims([GO_INDEX], 0)

  for t in range(max_length_targ):
    predictions, dec_hidden = decoder(dec_input,dec_hidden,enc_out)

    predicted_id = tf.argmax(predictions[0]).numpy()

    result.append(predicted_id)

    if targ_lang.index_word[predicted_id] == '<end>':
      return result, sentence, attention_plot

    dec_input = tf.expand_dims([predicted_id], 0)

  return result

from nltk.translate.bleu_score import sentence_bleu
scores = [sentence_bleu([target_index_test[i]],translate(source_index_test[i])) for i in range(source_index_test)]
print("Avg BLEU:",sum(scores)/len(scores))

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import matplotlib.pyplot as plt
#plt.hist([i for i in chn_vocab_word.values() if i<1000 and i>10])
plt.plot(loss_record)

