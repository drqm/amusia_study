import mne
from sys import argv

def export_free_listening(subject = 2, block = 'piazzolla',
    data_path = 'C:/Users/dariq/Dropbox/PC/Documents/projects/amusia_study/data/raw/',
    ica_path = 'C:/Users/dariq/Dropbox/PC/Documents/projects/amusia_study/data/ICA/',
    out_dir = 'C:/Users/dariq/Dropbox/PC/Documents/projects/amusia_study/data/free_listening/data/'):

    file_name = data_path + subject + '_' + block + '.bdf'
    ica_fname = ica_path + subject + '_' + block + '-ica.fif'
    out_fname = out_dir + subject + '_' + block + '_ica-raw.edf'
    raw = mne.io.read_raw_bdf(file_name,preload=True)
    ica = mne.preprocessing.read_ica(ica_fname)
    new_ch_names = {s: (s[2:] if s[0] == '1' else s) for s in raw.info['ch_names']}
    raw.rename_channels(new_ch_names);
    raw = ica.apply(raw)
    print('exporting to ', out_fname)
    raw.export(out_fname,fmt='edf')
