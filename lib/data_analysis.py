import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from data_processor import DataProcessor

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

# This script is used for data analysis

df = DataProcessor('source.csv').train_df
df["volume"] = df["volume"].pct_change()
df = df.iloc[1:]


signal = tf.constant(df["volume"], dtype=tf.float32)
signal_mean = tf.reduce_mean(signal)
signal_without_dc = signal - signal_mean
windowed_signal = signal_without_dc * tf.signal.hamming_window(len(signal))

fft = tf.signal.rfft(windowed_signal)
fft_magnitudes = tf.abs(fft)
main_frequency_index = tf.argmax(fft_magnitudes)
frequency = main_frequency_index / len(signal)
main_frequency_magnitude = fft_magnitudes[main_frequency_index]
total_energy = tf.reduce_sum(tf.square(fft_magnitudes))
energy_ratio = (main_frequency_magnitude ** 2) / total_energy

print("Main Frequency:", frequency.numpy(), "Hz")
print("Main Frequency Magnitude:", main_frequency_magnitude.numpy())
print("Energy Ratio of Main Frequency:", energy_ratio.numpy())

f_per_dataset = np.arange(0, len(fft))

n_samples_h = len(signal)/60
hours_per_year = 24*365.2524
years_per_dataset = n_samples_h/(hours_per_year)

f_per_year = f_per_dataset/years_per_dataset
plt.step(f_per_year, np.abs(fft))
plt.xscale('log')
plt.ylim(0, 40000)
plt.xlim([0.1, max(plt.xlim())])
plt.xticks([1, 365.2524], labels=['1/Year', '1/day'])
plt.xlabel('Frequency (log scale)')
plt.show()