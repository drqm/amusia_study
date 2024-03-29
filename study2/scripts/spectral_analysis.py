# This script produces waveform, spectrogram and autocorrelation plots for the stimuli
# used in the study. It also calculates the spectral entropy of the sitmuli.
# The stimuli are stored in the 'stimuli' folder and are
# available upon request from the authors.

# You can run the script as is or you can also run it interactively using
# VSCode or something similar. This is what '# %%' lines are for.

# %%
import numpy as np
import librosa # for general stft/spectrogram plotting
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import antropy as ant # for spectral entropy

# %%
# load sounds
hh, sr = librosa.load('../stimuli/hh_spectr.wav', sr=None, mono=False)
piano, sr = librosa.load('../stimuli/piano_spectr.wav', sr=None, mono=False)

# hh[0] holds left, hh[1] right channels

# %%

## WAVEFORM PLOTS

fig, (ax, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(14, 4), dpi=300)
ax.set_title("Hi-hat")
librosa.display.waveshow(hh[0][:int(sr/4)], sr=sr, ax=ax, marker='.')
ax2.set_title("Piano")
librosa.display.waveshow(piano[0][:int(sr/4)], sr=sr, ax=ax2, marker='.')
plt.savefig('waveforms.pdf')
plt.savefig('waveforms.png')



# %%
## SPECTROGRAM

# STFT
# hihat
# left channel
D_hh_l = librosa.stft(hh[0], hop_length=128, n_fft=4096)  # STFT of y

# right channel
D_hh_r = librosa.stft(hh[1], hop_length=128, n_fft=4096)  # STFT of y

# piano
# left channel
D_piano_l = librosa.stft(piano[0], hop_length=128, n_fft=4096)  # STFT of y

# right channel
D_piano_r = librosa.stft(piano[1], hop_length=128, n_fft=4096)  # STFT of y

# determine max amp for all four signals
S_max = np.max([D_hh_l, D_hh_r, D_piano_r, D_piano_l])

# Rescale amplitudes to db
# Reference is piano, right channel
S_l_hh = librosa.amplitude_to_db(np.abs(D_hh_l), ref=S_max)
S_r_hh = librosa.amplitude_to_db(np.abs(D_hh_r), ref=S_max)
S_l_piano = librosa.amplitude_to_db(np.abs(D_piano_l), ref=S_max)
S_r_piano = librosa.amplitude_to_db(
    np.abs(D_piano_r), ref=S_max)


# %%

# a stereo spectrogram of hihat and piano


fig, axs = plt.subplots(4, figsize=(14, 12), dpi=300)
plt.subplots_adjust(hspace=.3)

# plot spectrograms
librosa.display.specshow(S_l_hh, sr=sr, y_axis='log',
                         x_axis='s', hop_length=128, vmax=0, vmin=-80,
                         cmap='viridis', ax=axs[0])

librosa.display.specshow(S_r_hh, sr=sr, y_axis='log',
                         x_axis='s', hop_length=128, vmax=0, vmin=-80,
                         cmap='viridis', ax=axs[1])

librosa.display.specshow(S_l_piano, sr=sr, y_axis='log',
                         x_axis='s', hop_length=128, vmax=0, vmin=-80,
                         cmap='viridis', ax=axs[2])

librosa.display.specshow(S_r_piano, sr=sr, y_axis='log',
                         x_axis='s', hop_length=128, vmax=0, vmin=-80,
                         cmap='viridis', ax=axs[3])


# define axis titles, ticks and tick labels for y axis
ticks = [60, 120, 250, 500, 1000, 2000, 4000, 8000, 16000]
labels = ['60', '120', '250', '500', '1k', '2k', '4k', '8k', '16k']
titles = ['Hihat (L)', 'Hihat (R)', 'Piano (L)', 'Piano (R)']

# set
for ax, title in zip(axs, titles):
    ax.set(title=title, xlabel=None)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)


# add rectangles to axes
for ax in axs:
    # intensity
    ax.add_patch(Rectangle(
        (.22, 20), width=.22, height=18000,
        label='Standard',  linewidth=1,
        edgecolor='r', facecolor='none'
    ))
    ax.text(.28, -70, 'intensity', color='red')

    # move everything on x by 0.5
    offset = .5
    # location
    ax.add_patch(Rectangle(
        (.22+offset, 20), width=.22, height=18000,
        label='Standard',  linewidth=1,
        edgecolor='r', facecolor='none'
    ))
    ax.text(.28+offset, -70, 'location', color='red')

    # timbre
    ax.add_patch(Rectangle(
        (.22+(2*offset), 20), width=.22, height=18000,
        label='Standard',  linewidth=1,
        edgecolor='r', facecolor='none'
    ))
    ax.text(.3+(2*offset), -70, 'timbre', color='red')

# xlabel only for last axis
axs[3].set(xlabel='Time (s)')

# show plot
plt.savefig('spectrogram.png')
plt.savefig('spectrogram.jpg')
plt.show()



# %%

## AUTOCORRELATION PLOT

def sec_smpls(sec, sr=44100):
    return int(sr * sec)


def norm(x):
    return x / np.max(x)


# choose a time window for autocorrelation - 10ms
time_window = sec_smpls(.01)

# cut the first (standard) note from the piano file
piano_std = librosa.util.normalize(piano[0][:sec_smpls(.1)])

# same for hihat
hh_std = librosa.util.normalize(hh[0][:sec_smpls(.1)])

# make some (white) noise
noise = np.random.uniform(low=-1, high=1, size=sec_smpls(.1))

# compute autocorrelations
ac_piano = librosa.autocorrelate(piano_std, max_size=time_window)
ac_hh = librosa.autocorrelate(hh_std, max_size=time_window)
ac_noise = librosa.autocorrelate(noise, max_size=time_window)

# compute the x axis in ms
time_ms = librosa.samples_to_time(np.arange(time_window), sr=sr) * 1000

# plot
plt.figure(2, figsize=(14, 4), dpi=150)
plt.suptitle('Normalized auto-correlations')

plt.subplot(131)
ax = plt.gca()
ax.plot(time_ms, norm(ac_piano))
ax.set(title='Piano', xlabel='Time (ms)')
plt.ylim(-1, 1)

plt.subplot(132)
ax = plt.gca()
ax.plot(time_ms, norm(ac_hh))
ax.set(title='Hihat', xlabel='Time (ms)')
plt.ylim(-1, 1)

plt.subplot(133)
ax = plt.gca()
ax.plot(time_ms, norm(ac_noise))
ax.set(title='White noise', xlabel='Time (ms)')
plt.ylim(-1, 1)


# uncomment this if you want to show the plot instead of writing it to file
# plt.show()

plt.savefig('../results/autocorrelations.png')
plt.savefig('../results/autocorrelations.pdf')
print('Saved autocorrelation plots to ../results/')

# %%
## SPECTRAL ENTROPIES
# print out spectral entropies

# spectral entropy of the hihat signal
ent_hh = ant.spectral_entropy(hh_std, sf=sr)

# spectral entropy of the piano signal
ent_piano = ant.spectral_entropy(piano_std, sf=sr)

# spectral entropy of white noise
ent_noise = ant.spectral_entropy(noise, sf=sr)

print(f'Spectral entropy of hihat signal is {ent_hh}')
print(f'Spectral entropy of piano signal is {ent_piano}')
print(f'Spectral entropy of noise signal is {ent_noise}')


# %%
