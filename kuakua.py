import tensorflow as tf
path_to_corpus = tf.keras.utils.get_file('/content/corpus.txt',\
                                         'https://github.com/xiaopangxia/kuakua_corpus/raw/master/douban_kuakua_qa.txt')

import re
def clean_chn(text):
    text = re.sub(r"[^\u2E80-\uFE4F\uFF00-\uFFEF]", "", text)
    return text

questions = []
answers = []
MAX_LENGTH = 20


with open('corpus.txt') as f:
  lines = f.readlines()
  for i in range(len(lines)//2):
    line1 = lines[i].split()
    if len(line1) < 2 or '谢' in line1[1]:
      continue
    line1 = list(clean_chn(line1[1]))
    line2 = lines[i+1].split()
    if len(line2) < 2 or '谢' in line2[1]:
      continue
    line2 = list(clean_chn(line2[1]))
    if len(line1) > MAX_LENGTH or len(line2) > MAX_LENGTH:
      continue
    questions.append(line1)
    answers.append(line2)

print('Total:',len(questions))

def build_vocab(lines):
  vocab = {}
  for line in lines:
    for word in line:
        if word not in vocab:
            vocab[word] = 1
        else:
            vocab[word] += 1
  return vocab

vocab = build_vocab(questions+answers)

print("Vocab:",len(vocab))

GO_INDEX = 3
VOCAB_SIZE = 2000 + 4

codes = ['<PAD>','<EOS>','<UNK>','<GO>']
int_to_word = codes + sorted(vocab.keys(), key = lambda k:-vocab[k])[:VOCAB_SIZE - 4]
word_to_int = dict(zip(int_to_word,range(VOCAB_SIZE)))

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

source_lines = [['<GO>']+line+['<EOS>'] for line in questions]
source_index = build_index(source_lines,int_to_word,word_to_int)

target_lines = [['<GO>']+line+['<EOS>'] for line in answers]
target_index = build_index(target_lines,int_to_word,word_to_int)

VOCAB_SIZE = len(int_to_word)
EMBEDDING_DIM = 256
UNITS = 1024
MAX_LENGTH = MAX_LENGTH + 2
BUFFER_SIZE = len(source_index)
BATCH_SIZE = 64

input_tensor = tf.keras.preprocessing.sequence.pad_sequences(source_index,maxlen=MAX_LENGTH,padding='post')
target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_index,maxlen=MAX_LENGTH,padding='post')
dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

example_input_batch, example_target_batch = next(iter(dataset))
example_input_batch.shape, example_target_batch.shape

share_embedding = tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
encoder_gru = tf.keras.layers.GRU(UNITS,return_sequences=True,return_state=True,recurrent_initializer='glorot_uniform')

encoder_hidden = tf.keras.Input(shape=(UNITS,))
encoder_input = tf.keras.Input(shape=(MAX_LENGTH,))

encoder_embed = share_embedding(encoder_input)
encoder_output, encoder_state = encoder_gru(encoder_embed, initial_state = encoder_hidden)

encoder = tf.keras.Model(inputs=[encoder_input,encoder_hidden],outputs=[encoder_output, encoder_state])
encoder.summary()

sample_hidden = tf.zeros((BATCH_SIZE, UNITS))
sample_output, sample_hidden = encoder((example_input_batch, sample_hidden))
print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
print ('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))

decoder_gru = tf.keras.layers.GRU(UNITS,return_sequences=True,return_state=True,recurrent_initializer='glorot_uniform')
decoder_fc = tf.keras.layers.Dense(VOCAB_SIZE)
decoder_attention = tf.keras.layers.AdditiveAttention()

decoder_input = tf.keras.Input(shape=(1,))
decoder_enchidden = tf.keras.Input(shape=(UNITS,))
decoder_encoutput = tf.keras.Input(shape=(MAX_LENGTH,UNITS))

decoder_context = decoder_attention([tf.expand_dims(decoder_enchidden, 1), decoder_encoutput])
decoder_embed = share_embedding(decoder_input)
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
#checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
loss_record = []

import time,tqdm
EPOCHS = 5
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

def evaluate(sentence):
  sentence = ['<GO>']+list(sentence)+['<EOS>']
  inputs = [word_to_int[i] for i in sentence if i in int_to_word]
  inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],maxlen=MAX_LENGTH,padding='post')
  inputs = tf.convert_to_tensor(inputs)
  result = ''
  hidden = [tf.zeros((1, UNITS))]
  enc_out, enc_hidden = encoder((inputs, hidden))
  dec_hidden = enc_hidden
  dec_input = tf.expand_dims([GO_INDEX], 0)
  for t in range(MAX_LENGTH):
    predictions, dec_hidden = decoder((dec_input,dec_hidden,enc_out))
    predicted_id = tf.argmax(predictions[0]).numpy()
    result += int_to_word[predicted_id]
    if int_to_word[predicted_id] == '<EOS>':
      return result, sentence
    dec_input = tf.expand_dims([predicted_id], 0)
  return result, sentence

while True:
  sentence=input("来夸夸你: ")
  if sentence == '不要':
    break
  result,sentence = evaluate(sentence)
  print("哈哈哈：",result)


import matplotlib.pyplot as plt
#plt.hist([i for i in vocab.values() if i < 1000])
plt.plot(loss_record)

