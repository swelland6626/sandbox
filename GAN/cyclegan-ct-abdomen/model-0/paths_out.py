import pickle

filename = '/cvib2/apps/personal/swelland/sandbox/gan/cyclegan-ct-abdomen/model-0/pre_slices_paths.pkl'

open_file = open(filename, 'rb')
loaded_list = pickle.load(open_file)
open_file.close()

print(loaded_list)