from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.callbacks import Callback
import numpy as np
import keras.backend as K
import configuration as config
import models
import h5py

def fmeasure(y_true, y_pred):
    
    def precision(y_true, y_pred):
        true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
        predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)))
        precision = true_positives/(predicted_positives+ K.epsilon())
        return precision
    
    def recall(y_true, y_pred):
        true_positives = np.sum(np.round(np.clip(y_true*y_pred, 0, 1)))
        possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)))
    
        recall = true_positives/ (possible_positives + K.epsilon())
    
        return recall
    
    def fbeta_score(y_true, y_pred, beta = 1):
        if beta < 0:
            raise ValueError('The lowest beta value is 0')
        if np.sum(np.round(np.clip(y_true, 0, 1))) == 0:
            return 0
    
        p = precision(y_true, y_pred)
        r = recall(y_true, y_pred)
    
        bb = beta ** 2
    
        fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    
        return fbeta_score
    
    return fbeta_score(y_true, y_pred, beta = 1)


class PreProcessing:
		
	def loadData(self):   
		print("loading data...")
		data_src = config.data_src
		text_zero = open(data_src[0],"r", encoding = "ISO-8859-1").readlines()
		text_one = open(data_src[1],"r", encoding = "ISO-8859-1").readlines()
		text_two = open(data_src[2],"r", encoding = "ISO-8859-1").readlines()
		text_three = open(data_src[3],"r", encoding = "ISO-8859-1").readlines()
		text_four = open(data_src[4],"r", encoding = "ISO-8859-1").readlines()
		labels_one = [1]*len(text_one)
		labels_two = [2]*len(text_two)
		labels_three = [3]*len(text_three)
		labels_four = [4]*len(text_four)
		labels_zero = [0]*len(text_zero)
		texts = text_zero
		texts.extend(text_one)
		texts.extend(text_two)
		texts.extend(text_three)
		texts.extend(text_four)
		labels = labels_zero
		labels.extend(labels_one)
		labels.extend(labels_two)
		labels.extend(labels_three)
		labels.extend(labels_four)
		
		   
		tokenizer = Tokenizer(num_words=config.MAX_NB_WORDS)
		tokenizer.fit_on_texts(texts)
		sequences = tokenizer.texts_to_sequences(texts)

		word_index = tokenizer.word_index
		print('Found %s unique tokens.' % len(word_index))

		data = pad_sequences(sequences, maxlen=config.MAX_SEQUENCE_LENGTH)

		labels = np_utils.to_categorical(np.asarray(labels))
		print('Shape of data tensor:', data.shape)
		print('Shape of label tensor:', labels.shape)

		# split the data into a training set and a validation set
		indices = np.arange(data.shape[0])
		np.random.shuffle(indices)
		data = data[indices]
		labels = labels[indices]
		nb_validation_samples = int(config.VALIDATION_SPLIT * data.shape[0])
		nb_test_samples = int(config.TEST_SPLIT * data.shape[0])

		self.x_train = data[:-nb_validation_samples-nb_test_samples]
		self.y_train = labels[:-nb_validation_samples-nb_test_samples]
		self.x_val = data[-nb_validation_samples-nb_test_samples:-nb_test_samples]
		self.y_val = labels[-nb_validation_samples-nb_test_samples:-nb_test_samples]
		self.x_test = data[-nb_test_samples:]
		self.y_test = labels[-nb_test_samples:]
		self.word_index = word_index

	def loadEmbeddings(self):
		embeddings_src = config.embeddings_src
		word_index = self.word_index
		embeddings_index = {}
		f = open(embeddings_src, encoding = "UTF-8")
		i=0
		for line in f:
			values = line.split()
			word = values[0]
			coefs = np.asarray(values[1:], dtype='float32')
			embeddings_index[word] = coefs
			EMBEDDING_DIM = len(coefs)
			i+=1
			if i>10000:
				break
		f.close()

		print('Found %s word vectors.' % len(embeddings_index))
		embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
		for word, i in word_index.items():
			embedding_vector = embeddings_index.get(word)
			if embedding_vector is not None:
				# words not found in embedding index will be all-zeros.
				embedding_matrix[i] = embedding_vector
		self.embedding_matrix = embedding_matrix
		self.EMBEDDING_DIM = EMBEDDING_DIM
		self.MAX_SEQUENCE_LENGTH = config.MAX_SEQUENCE_LENGTH
		

def main():
	preprocessing = PreProcessing()
	preprocessing.loadData()
	preprocessing.loadEmbeddings()
	
	#cnn_model = models.CNNModel()
	cnn_model = models.CNNModel()
	params_obj = config.Params()
	
	# Establish params
	params_obj.num_classes=5
	params_obj.vocab_size = len(preprocessing.word_index)
	params_obj.inp_length = preprocessing.MAX_SEQUENCE_LENGTH
	params_obj.embeddings_dim = preprocessing.EMBEDDING_DIM
            
	# get model
	#model = cnn_model.getModel(params_obj=params_obj, weight=preprocessing.embedding_matrix)
	model = cnn_model.getModel(params_obj=params_obj, weight=preprocessing.embedding_matrix)
	x_train, y_train, x_val, y_val = preprocessing.x_train, preprocessing.y_train, preprocessing.x_val, preprocessing.y_val

	# train
	model.fit(x_train, y_train, validation_data=(x_val, y_val),
              epochs=params_obj.num_epochs, batch_size=params_obj.batch_size)
              
	#evaluate
	x_test, y_test = preprocessing.x_test, preprocessing.y_test
	
	scores = model.evaluate(x_test, y_test)

#	model.save('cnn-model-final.h5')
	
	y_pred = model.predict_on_batch(x_test)
	
	print('Accuracy : ', scores[1]*100)
	print('F-Score : ',fmeasure(y_test, y_pred))

if __name__ == "__main__":
	main()
