import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from os import path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.signal import find_peaks
from scipy.signal import welch



stimulus_desc_file = pd.read_excel('ECG_GSR_Emotions/Stimulus_Description.xlsx')
stimulus_desc_file.to_csv('ECG_GSR_Emotions/Stimulus_Description.csv', index=None, header=True)
stimulus_desc = pd.read_csv('ECG_GSR_Emotions/Stimulus_Description.csv')
print(stimulus_desc.head())

self_annotation_multimodal_file = pd.read_excel('ECG_GSR_Emotions/Self-Annotation Labels/Self-annotation Multimodal_Use.xlsx')
self_annotation_multimodal_file.to_csv('ECG_GSR_Emotions/Self-Annotation Labels/Self-annotation Multimodal_Use.csv', index=None, header=True)
self_annotation_multimodal = pd.read_csv('ECG_GSR_Emotions/Self-Annotation Labels/Self-annotation Multimodal_Use.csv')
self_annotation_multimodal['annotation'] = 'M'
self_annotation_multimodal.rename(columns={'V_Label':'Valence', 'A_Label':'Arousal','Four_Labels':'Four_Label'}, inplace=True)
print(self_annotation_multimodal.head())

self_annotation_singlemodal_file = pd.read_excel('ECG_GSR_Emotions/Self-Annotation Labels/Self-annotation Single modal_Use.xlsx')
self_annotation_singlemodal_file.to_csv('ECG_GSR_Emotions/Self-Annotation Labels/Self-annotation Single modal_Use.csv', index=None, header=True)
self_annotation_singlemodal = pd.read_csv('ECG_GSR_Emotions/Self-Annotation Labels/Self-annotation Single modal_Use.csv')
self_annotation_singlemodal['annotation'] = 'S'
self_annotation_singlemodal.rename(columns={'Male':'Gender', 'Session Id':'Session ID','Video Id':'Video ID'}, inplace=True)

self_annotation_frames = [self_annotation_multimodal, self_annotation_singlemodal]
merged_dataframe = pd.concat(self_annotation_frames)
table_frame = merged_dataframe.copy()

cols = []
cols.append('Raw Data')
for col in merged_dataframe.columns:
    cols.append(col)

# print(merged_dataframe.head())

arr_shape = 5000
raw_data_arr = []

def form_data(data_array = [], annotation = '', data_path = ''):
   
    for filename in os.listdir(data_path):
            data = np.loadtxt(os.path.join(data_path, filename), delimiter=',')
            data = data[0:arr_shape]
            filenames = filename.split('ECGdata_')[1]
            filenames = filenames.split('.dat')[0]
            filenames = filenames.lower()
            s = filenames.split('s')[1].split('p');
            p = s[1].split('v')
            s = s[0]
            v = p[1]
            p = p[0]
            data_row = merged_dataframe.loc[(merged_dataframe['Session ID'] == int(s)) & 
                             (merged_dataframe['Participant Id'] == int(p)) & 
                             (merged_dataframe['Video ID'] == int(v)) &
                             (merged_dataframe['annotation'] == 'M')]
            stim_row = stimulus_desc.loc[(stimulus_desc['Session ID'] == int(s)) & 
                             (stimulus_desc['Video ID'] == int(v))]
            for index, row in data_row.iterrows():
              data_array.append([data,
                                   row['Participant Id'], row['Session ID'], row['Video ID'],
                                   row['Name'], row['Age'], row['Gender'], row['Valence level'],
                                   row['Arousal level'], row['Dominance level'], row['Happy'],
                                   row['Sad'], row['Fear'], row['Anger'], row['Neutral'],
                                   row['Disgust'], row['Surprised'], row['Familiarity Score'],
                                   row['Emotion'], row['Valence'], row['Arousal'], row['Four_Label'],
                                   row['annotation'],  stim_row['Target Emotion'].iat[0]
                                   ])
    return data_array

   
raw_data_arr =  form_data(data_array = raw_data_arr, annotation = 'M', data_path = "ECG_GSR_Emotions/Raw Data/Multimodal/ECG/")
raw_data_arr =  form_data(data_array = raw_data_arr, annotation = 'S', data_path = "ECG_GSR_Emotions/Raw Data/Single Modal/ECG/")
cols.append('Target Emotion')
raw_dataframe = pd.DataFrame(raw_data_arr, columns = cols)
raw_dataframe.rename(columns = {'Participant Id':'Participant ID', 'annotation':'Modal', 'Four_Label':'Four label'}, inplace = True)
raw_dataframe['Familiarity Score'] = raw_dataframe['Familiarity Score'].fillna('Never watched')
raw_dataframe = raw_dataframe.replace(np.nan, '', regex=True)
print(raw_dataframe.head())

plot_frame = raw_dataframe.copy()
plot_frame.head()


plot_frame = plot_frame.drop(['Participant ID', 'Session ID', 'Familiarity Score', 'Age', 'Gender', 'Name'], axis = 1)
sad_data = plot_frame.loc[(plot_frame['Emotion'] == 'Sad') & (plot_frame['Target Emotion'] == 'sad')]
fear_data = plot_frame.loc[(plot_frame['Emotion'] == 'Fear')  & (plot_frame['Target Emotion'] == 'fear')]
happy_data = plot_frame.loc[(plot_frame['Emotion'] == 'Happy') & (plot_frame['Target Emotion'] == 'happy')]
anger_data = plot_frame.loc[(plot_frame['Emotion'] == 'Anger') & (plot_frame['Target Emotion'] == 'anger')]
neutral_data = plot_frame.loc[(plot_frame['Emotion'] == 'Neutral') & (plot_frame['Target Emotion'] == 'neutral')]
mixed_data = plot_frame.loc[(plot_frame['Emotion'] == 'Mixed') & (plot_frame['Target Emotion'] == 'neutral')]
disgust_data = plot_frame.loc[(plot_frame['Emotion'] == 'Disgust') & (plot_frame['Target Emotion'] == 'disgust')]
surprised_data = plot_frame.loc[(plot_frame['Emotion'] == 'Surprise') & (plot_frame['Target Emotion'] == 'surprise')]
def plot_signals(data_arr, title = ''):
    plt.clf()
    plt.figure(figsize=(12, 4))
   
    for index, row in data_arr.iterrows():
        y = row['Raw Data']
        plt.plot(y)
        #x = np.arange(y.size)
        #plt.plot(x, y)
   
    plt.tight_layout()
    plt.title(title)
    plt.show()

# plot_signals(data_arr = plot_frame['Valence'], title = 'Valence')
# plt.show()

# print(self_annotation_singlemodal_file.shape)
# for index, row in sad_data.iterrows():
#     valence = row['Valence']
#     plt.plot(valence)

# Define a function to extract ECG features

# Define a function to extract ECG features
# Define a function to extract PSD features
def extract_psd_features(ecg_data, sampling_rate):
    try:
        # Ensure ecg_data is a 1D NumPy array and contains only numeric values
        ecg_data = np.array(ecg_data, dtype=np.float64)

        # Compute the power spectral density using Welch's method
        f, psd = welch(ecg_data, fs=sampling_rate, nperseg=1024)

        # Define frequency bands (you can adjust these as needed)
        # Example: Low Frequency (LF) is typically 0.04-0.15 Hz, and High Frequency (HF) is 0.15-0.4 Hz
        lf_band = (0.04, 0.15)
        hf_band = (0.15, 0.4)

        # Calculate the power within LF and HF bands
        lf_power = np.trapz(psd[(f >= lf_band[0]) & (f <= lf_band[1])])
        hf_power = np.trapz(psd[(f >= hf_band[0]) & (f <= hf_band[1])])

        # Return the PSD features as a dictionary
        features = {
            'LF_Power': lf_power,
            'HF_Power': hf_power
        }

        return features

    except Exception as e:
        print("Error:", str(e))
        return None

# Sample rate of the ECG data (replace with your actual sampling rate)
sampling_rate = 1000  # Example: 1000 samples per second

# Assuming you have a DataFrame called 'features_df' containing your other features
# and you want to calculate PSD features for each sample in your dataset
psd_features_list = []  # To store the calculated PSD features for each sample

# for ecg_data in features_df['Raw Data']:
#     psd_features = extract_psd_features(ecg_data, sampling_rate)
#     if psd_features is not None:
#         psd_features_list.append(psd_features)
#     else:
#         continue
#         # Handle the case where feature extraction fails for a sample
#         # You can choose to skip the sample or handle it based on your needs

# # Convert the list of PSD features into a DataFrame
# psd_features_df = pd.DataFrame(psd_features_list)


valence_df = raw_dataframe[['Raw Data','Valence level']]
valence_df['ECG_mean'] = valence_df['Raw Data'].apply(lambda seq: np.mean(seq))
print(valence_df.head())
# print(valence_df.head())
ecg_data = valence_df['Raw Data'].values  # Assuming 'ECG_mean' contains ECG data
valence_levels = valence_df['Valence level'].values  # Assuming 'Valence level' contains emotion labels

# Define the sampling rate of your ECG data (replace with your actual sampling rate)
sampling_rate = 1000  # Example: 1000 samples per second

# Initialize lists to store extracted features
dominant_frequencies = []

# Loop through each ECG sequence and extract dominant frequency components
for seq in ecg_data:
    # Perform FFT
    fft_result = np.fft.fft(seq)
    frequencies = np.fft.fftfreq(len(seq), 1 / sampling_rate)

    # Calculate the power spectrum
    power_spectrum = np.abs(fft_result) ** 2

    # Find peaks in the power spectrum
    peaks, _ = find_peaks(power_spectrum, height=0.01)  # Adjust the threshold as needed

    # Extract the dominant frequency components
    dominant_freqs = frequencies[peaks]

    # You can choose to extract other features from the frequency domain here
    # For example, you can compute the power or amplitude at specific frequency bands
   
    # Append the dominant frequency components to the list
    dominant_frequencies.append(dominant_freqs)

# Create a new DataFrame to store the extracted features
features_df = pd.DataFrame({
    'Valence level': valence_levels,
    'Dominant Frequencies': dominant_frequencies,
    'ECG_mean': valence_df['ECG_mean']
})
features_df['Dominant Frequencies'] = features_df['Dominant Frequencies'].apply(lambda x: ' '.join(map(str, x)).split())

# Ensure that each element is now a list of space-separated strings
features_df['Dominant Frequencies'] = features_df['Dominant Frequencies'].apply(lambda x: list(map(float, x)))

# Calculate the maximum frequency from the list of dominant frequencies
features_df['Max Frequency'] = features_df['Dominant Frequencies'].apply(lambda x: max(x))
X = features_df[['Max Frequency','ECG_mean']]  # Features
y = valence_df['Valence level']  # Target variable

# Split the dataset into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=450)

# Create and train a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=500, random_state=450)  # You can adjust hyperparameters
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

print(features_df.head())