import numpy as np
import math
import librosa
import librosa.display

from tkinter import ttk
from tkinter import *

import tkinter as tk
from tkinter import filedialog, messagebox

file_path_ideal = None
file_path_test = None

def select_ideal_file():
    global file_path_ideal
    file_path_ideal = filedialog.askopenfilename(
        title="Select an Audio File",
        filetypes=[("Audio Files", "*.mp3 *.wav"), ("All Files", "*.*")]
    )
    if file_path_ideal:
        messagebox.showinfo("Selected File", f"You selected: {file_path_ideal}")
    else:
        messagebox.showinfo("No File Selected", "Please select a file to analyze.")

def select_test_file():
    global file_path_test
    file_path_test = filedialog.askopenfilename(
        title="Select an Audio File",
        filetypes=[("Audio Files", "*.mp3 *.wav"), ("All Files", "*.*")]
    )
    if file_path_test:
        messagebox.showinfo("Selected File", f"You selected: {file_path_test}")
    else:
        messagebox.showinfo("No File Selected", "Please select a file to analyze.")

#extracts the dominant frequency present at each frame of an audio file
def extract_frequency(audio):
        
    y, sr = librosa.load(audio)

    n_fft = 2048
    stft_result = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=None))
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    #stft_result = np.abs(librosa.stft(y, hop_length=None))
    #frequencies = librosa.core.fft_frequencies(sr=sr, n_fft=stft_result.shape[0] * 2 - 1)

    dom_freq = []
    for frame in stft_result.T:

        index = np.argmax(frame)
        dominant_freq = frequencies[index]
        dom_freq.append(dominant_freq)
    #for frame in stft_result.T: #green text works the same as current code, just wanted to experiment
            #index = np.argmax(frame)  
            #dom_freq.append(frequencies[index])

    return dom_freq

#given a frequency, this function returns a corresponding note and octave 
def freq_to_note(freq):
    notes = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']

    note_number = 12 * math.log2(freq / 440) + 49  
    note_number = round(note_number)
        
    note = (note_number - 1 ) % len(notes)
    note = notes[note]
    
    octave = (note_number + 8 ) // len(notes)

    return note#, octave

#calculates the levenshtein distance between two strings, to provide accuracy measurement while accounting for misalignments
def levenshtein_distance(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[n][m]

#removes frequencies that are not within the bounds of the sounds a piano could make
def remove_empty_space(frequencies):
    
    to_be_removed = []
    for i in range(0, len(frequencies)):
        freq = frequencies[i]
        if (freq < 27.5 or freq > 4168):
            to_be_removed.append(i)

    x = 0
    for i in range(0, len(to_be_removed)):
        delete = to_be_removed[i] - x
        del frequencies[delete]
        x = x + 1
    
    return (frequencies)

#creates the array of notes, with each note being calculated at the onset of a new sound, indicating a new note has been played
def create_notes_arr(frequencies, audio):
    notes = []
    y, sr = librosa.load(audio)
    onset_env = librosa.onset.onset_strength(y=y)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env)

    to_be_removed = []
    for i in range(0, len(frequencies)):
        freq = frequencies[i]
        if (freq < 27.5 or freq > 4168):
            to_be_removed.append(i)

    x = 0
    for i in range(0, len(to_be_removed)):
        comparison = to_be_removed[i]
        for u in range(0, len(onset_frames)):
            if (comparison == onset_frames[u - x]):
                onset_frames = np.delete(onset_frames, u - x)
                x = x + 1

    for i in range(0, len(to_be_removed)):
        comparison = to_be_removed[i]
        for u in range(0, len(onset_frames)):
            if (onset_frames[u] >= comparison):
                onset_frames[u] = onset_frames[u] - 1

    x = 0
    for i in range(0, len(onset_frames)):
        if onset_frames[i] > len(frequencies):
            onset_frames = np.delete(onset_frames, i - x)
            x = x + 1

    x = 0
    for i in range(0, len(to_be_removed)):
        delete = to_be_removed[i] - x
        del frequencies[delete]
        x = x + 1

    x = 0
    for i in range(0, len(onset_frames)):
        if onset_frames[i - x] > len(frequencies):
            onset_frames = np.delete(onset_frames, i - x)
            x = x + 1

    onset_frames_to_be_removed = []
    for i in range(1, len(onset_frames)):
        if (onset_frames[i] - 10 <= onset_frames[i - 1]):
            onset_frames_to_be_removed.append(i)

    x = 0
    for i in range(0, len(onset_frames_to_be_removed)):
        onset_frames = np.delete(onset_frames, onset_frames_to_be_removed[i] - x)
        x = x + 1

    for i in range(0, len(onset_frames)):
        notes.append(frequencies[onset_frames[i]])

    for i in range(0, len(notes)):
        notes[i] = freq_to_note(notes[i])
        
    return (notes)

#calculates the accuracy of notes between two audio files
def get_note_accuracy(ideal, test):

    dom_freq_ideal = extract_frequency(ideal)
    dom_freq_test = extract_frequency(test)

    #dom_freq_ideal = remove_empty_space(dom_freq_ideal) #Removing empty spaces misaligns the frames of the frequency and onset_frames index
    #dom_freq_test = remove_empty_space(dom_freq_test)

    notes_ideal = create_notes_arr(dom_freq_ideal, ideal)
    notes_test = create_notes_arr(dom_freq_test, test)

    print(f'ideal: ', notes_ideal)
    print(f'test: ', notes_test)

    distance = levenshtein_distance(notes_ideal, notes_test)
    similarity = (1 - distance / max(len(notes_ideal), len(notes_test))) * 100
	
    return similarity 

#creates an array of the distance between onset frames in an audio file, which is used as a metric for tempo
def create_tempo_arr(audio):
    tempo = []
    y, sr = librosa.load(audio)
    onset_env = librosa.onset.onset_strength(y=y)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env)

    frames_to_be_removed = []
    for i in range(1, len(onset_frames)):
        if (onset_frames[i] - 10 <= onset_frames[i - 1]):
            frames_to_be_removed.append(i)

    x = 0
    for i in range(0, len(frames_to_be_removed)):
        onset_frames = np.delete(onset_frames, frames_to_be_removed[i] - x)
        x = x + 1

    for i in range(1, len(onset_frames)):
        distance = onset_frames[i] - onset_frames[i - 1]
        tempo.append(distance)
    
    return tempo

#calculates tempo accuracy between arrays and curves grade for slight misalignments
def get_tempo_accuracy(ideal, test):
    ideal_tempo = create_tempo_arr(ideal)
    test_tempo = create_tempo_arr(test)

    if (len(ideal_tempo) > len(test_tempo)):
        shortest = test_tempo
    else:
        shortest = ideal_tempo
    for i in range(0, len(shortest)): #applies grade curve for tempo if onset_frames are slightly off
        if (ideal_tempo[i] - 5 < test_tempo[i] or ideal_tempo[i] + 5 > test_tempo[i]):
            test_tempo[i] = ideal_tempo[i]

    distance = levenshtein_distance(ideal_tempo, test_tempo)
    similarity = (1 - distance / max(len(ideal_tempo), len(test_tempo))) * 100

    return similarity

def get_total_accuracy(ideal, test):
    global note_accuracy
    global tempo_accuracy
    global total_accuracy

    note_accuracy = get_note_accuracy(ideal, test)
    tempo_accuracy = get_tempo_accuracy(ideal, test)
    total_accuracy = note_accuracy * .5 + tempo_accuracy * .5
    
    print(f'Note Accuracy: {note_accuracy:.2f}%')
    print(f'Tempo Accuracy: {tempo_accuracy:.2f}%')
    print(f'Overall Accuracy: {total_accuracy:.2f}%')
    return note_accuracy, tempo_accuracy, total_accuracy

def close_window():
    root.destroy()

root = tk.Tk()
root.title("Audio File Analyzer")

ideal_audio = tk.Label(root, text = "Insert your audio file of the ideal performance.")
ideal_audio.pack(pady=30, padx=25)
select_button = tk.Button(root, text="Select Ideal Audio File", command=select_ideal_file)
select_button.pack(pady=29, padx=25)

ideal_audio = tk.Label(root, text = "Insert your audio file of the performance to be tested.")
ideal_audio.pack(pady=30, padx=100)
select_button = tk.Button(root, text="Select Test Audio File", command=select_test_file)
select_button.pack(pady=30, padx=100)

next_button = tk.Button(root, text="GRADE!", command=close_window)
next_button.pack(pady=10)

root.mainloop()

get_total_accuracy(file_path_ideal, file_path_test)

root = tk.Tk()
root.title("Audio File Analyzer")

label = tk.Label(root, text=f"Note Accuracy: {note_accuracy: .2f}%", font=("Helvetica", 16))
label.pack()
label = tk.Label(root, text=f"Tempo Accuracy: {tempo_accuracy: .2f}%", font=("Helvetica", 16))
label.pack()
label = tk.Label(root, text=f"Total Accuracy: {total_accuracy: .2f}%", font=("Helvetica", 16))
label.pack()

root.mainloop()
