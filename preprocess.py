import numpy as np
import os
import matplotlib.pyplot as plt
from config_training import config
import pandas as pd


def main():
	data_path = config['data_path']
	annos_path = config['annos_path']
	filenames = np.genfromtxt(config['filenames'], dtype=str)
	save_dir = config['preprocess_result_path']

	for file in filenames:
		data_file = os.path.join(data_path, '%s.jpg' % (file))
		annos_file = os.path.join(annos_path, '%s.xls' % (file))
		save_npy(file, data_file, annos_file, save_dir)

def save_npy(file, data_file, annos_file, save_dir):
	img = plt.imread(data_file)
	annos = pd.read_excel(annos_file)

	img = np.moveaxis(img, -1, 0)
	dx = [16 for _ in xrange(len(annos))]
	annos['dx'] = dx
	annos['dy'] = dx
	labels = annos.as_matrix(columns=['Cell_Y', 'Cell_X', 'dy', 'dx'])

	np.save(os.path.join(save_dir, '%s_img.npy' % (file)), img)
	np.save(os.path.join(save_dir, '%s_label.npy' % (file)), labels)
	print 'Finished processing ', file, ', image shape ', img.shape, ', total # of label ', len(labels)

if __name__ == '__main__':
	main()
