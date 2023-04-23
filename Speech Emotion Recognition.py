# Import required libraries
import os
import numpy as np
import pandas as pd
import seaborn as sns
import librosa
import librosa.display
import matplotlib.pyplot as plt
from IPython.display import Audio
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# path for the dataset
path = r'C:\Users\navee\Downloads\computer_vision\CV projects\speech_recognition\TESS Toronto emotional speech set data'

# Create empty lists to store audio files and their respective labels
all_images = []
labels = []

# Iterate through all files in the directory and subdirectories using os.walk() method
for (root, dirs, file) in os.walk(path):
    # Add the path of each audio file to the all_images list and its respective label to the labels list
    for f in file:
        all_images.append(os.path.join(root, f))
        label = f.split('_')[-1]
        label = label.split('.')[0]
        labels.append(label.lower())
        
        # The dataset consists of 7 different emotions, and each emotion has 400 audio files. 
        # So, after appending the path of 2800 audio files, exit the loop.
        if len(all_images) == 2800:
            break

print('Dataset is Loaded')

# Create a pandas dataframe to store the audio files and their respective labels
df = pd.DataFrame()
df['speech'] = all_images
df['label'] = labels

# Display the first 5 rows
df.head()

# Print the count of each label in the dataframe
df['label'].value_counts()

# Visualize the count of each label using a histogram
plt.figure(figsize = [8, 8])
sns.displot(df['label'])

# Define two helper functions to plot the waveform and the spectrogram of an audio file
def waveplot(data, sr, emotion):
    plt.figure(figsize = [10, 8])
    plt.title(emotion, size = 20)
    librosa.display.waveshow(data, sr = sr)
    plt.show()
    
def spectogram(data, sr, emotion):
    x = librosa.stft(data)
    xdb = librosa.amplitude_to_db(abs(x))
    plt.figure(figsize = [11, 4])
    plt.title(emotion, size = 20)
    librosa.display.specshow(xdb, sr = sr, x_axis = 'time', y_axis = 'hz')
    plt.colorbar()


# In[10]:

# Define a few emotions to display the waveform and spectrogram of some random audio files of each emotion
emotion = 'fear'
path = np.array(df['speech'][df['label']==emotion])[0]
data, sampling_rate = librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectogram(data, sampling_rate, emotion)
Audio(path)

emotion = 'angry'
path = np.array(df['speech'][df['label']==emotion])[5]
data, sampling_rate = librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectogram(data, sampling_rate, emotion)
Audio(path)

emotion = 'disgust'
path = np.array(df['speech'][df['label']==emotion])[8]
data, sampling_rate = librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectogram(data, sampling_rate, emotion)
Audio(path)

emotion = 'sad'
path = np.array(df['speech'][df['label']==emotion])[6]
data, sampling_rate = librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectogram(data, sampling_rate, emotion)
Audio(path)

emotion = 'ps'
path = np.array(df['speech'][df['label']==emotion])[1]
data, sampling_rate = librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectogram(data, sampling_rate, emotion)
Audio(path)

emotion = 'happy'
path = np.array(df['speech'][df['label']==emotion])[0]
data, sampling_rate = librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectogram(data, sampling_rate, emotion)
Audio(path)


# # Feature extraction
def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration = 3, offset = 0.5)
    mfcc = np.mean(librosa.feature.mfcc(y = y, sr = sr, n_mfcc = 40).T, axis = 0)
    return mfcc


# In[17]:


extract_mfcc(df['speech'][0])


# In[18]:


# def compute_spectrogram(audio_file, n_fft=2048, hop_length=512, sr=22050):
#     # load audio data
#     y, sr = librosa.load(audio_file, sr=sr)

#     # compute spectrogram
#     spec = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
#     spec_db = librosa.amplitude_to_db(abs(spec))

#     return spec_db, sr

# def plot_spectrogram(spec_db, sr, hop_length=512):
#     # visualize spectrogram
#     plt.figure(figsize=(12, 6))
#     librosa.display.specshow(spec_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
#     plt.colorbar(format='%+2.0f dB')
#     plt.title('Spectrogram')
#     plt.tight_layout()
#     plt.show()


# In[19]:


# # compute spectrogram
# spec_db, sr = compute_spectrogram(df['speech'][5])

# # plot spectrogram
# plot_spectrogram(spec_db, sr)


# In[20]:


# def extract_chroma_features(audio_file, sr=22050, n_fft=2048, hop_length=512):
#     # Load audio data
#     y, sr = librosa.load(audio_file, sr=sr)

#     # Compute chroma features
#     chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)

#     # Normalize chroma features
#     chroma_norm = librosa.util.normalize(chroma, norm=np.inf, axis=1)

#     return chroma_norm


# In[21]:


# # Extract chroma features
# chroma = extract_chroma_features(df['speech'][0])
# chroma


# In[23]:


X_mfcc = df['speech'].apply(lambda x: extract_mfcc(x))


# In[24]:


X = [x for x in X_mfcc]
X = np.array(X)
X.shape


# In[25]:


## input split
X = np.expand_dims(X, -1)
X.shape


# In[26]:


from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
y = enc.fit_transform(df[['label']])


# In[27]:


y = y.toarray()


# # Training LSTM Model

# In[28]:


from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from collections.abc import Iterable

model = Sequential([
    LSTM(256, return_sequences=False, input_shape=(40,1)),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(7, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
print(Iterable)


# In[35]:


from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
import tensorflow as tf

checkpoint_filepath = 'best_model.h5'
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1)


def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

lr_scheduler_callback = LearningRateScheduler(scheduler)

history = model.fit(X, y, validation_split=0.2, epochs=50, batch_size=64, callbacks=[model_checkpoint_callback, lr_scheduler_callback])


# # Plot the results

# In[36]:


epochs = list(range(50))
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, label='train accuracy')
plt.plot(epochs, val_acc, label='val accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()


# In[37]:


loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(epochs, loss, label='train loss')
plt.plot(epochs, val_loss, label='val loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


# In[ ]:




