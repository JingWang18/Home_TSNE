import numpy as np 
from matplotlib import pyplot as plt 
from sklearn.manifold import TSNE
import itertools
import os
import seaborn as sns

feature_s = np.load('AR_TSL3_feat_s_list_max.npy')
feature_t = np.load('AR_TSL3_feat_t_list_max.npy')
label_s = np.load('AR_TSL3_feat_s_label_list_max.npy')
label_t = np.load('AR_TSL3_feat_t_label_list_max.npy')

# feature_s_sourceonly = np.load('AR_sourceonly_feat_s_list_max.npy')
# feature_t_sourceonly = np.load('AR_sourceonly_feat_t_list_max.npy')
# label_s_sourceonly = np.load('AR_sourceonly_feat_s_label_list_max.npy')
# label_t_sourceonly = np.load('AR_sourceonly_feat_t_label_list_max.npy')

def draw_tsne(feature_s, feature_t, label_s, label_t, name = 'init'):
	feature_s_domain_label = np.zeros(np.size(feature_s,0))
	feature_t_domain_label = np.ones(np.size(feature_t,0))

	features = np.concatenate((feature_s,feature_t), axis = 0)
	labels = np.concatenate((label_s,label_t), axis = 0)
	domain_labels = np.concatenate((feature_s_domain_label,feature_t_domain_label), axis = 0)
	print(np.size(features,0), np.size(features,1))
	# features = np.reshape(features,(np.size(features,0), np.size(features,1)*np.size(features,2)*np.size(features,3)))
	# print(np.size(features,0), np.size(features,1))
	import random
	import time
	nums = [x for x in range(np.size(features,0))]
	random.shuffle(nums)
	random_samples = np.array(nums[0:5000])
	tic = time.time()
	tsne = TSNE(n_components=2, verbose=1, perplexity=40, init='pca', n_iter = 3000)
	# tsne_results = tsne.fit_transform(features[random_samples])
	tsne_results = tsne.fit_transform(features)
	toc = time.time()
	print('elasped time:', toc-tic)
	# plt.scatter(tsne_results[:,0], tsne_results[:,1], c=labels[random_samples], s=5)
	# plot_embedding(tsne_results, labels[random_samples], labels[random_samples], 'training_mode', name)
	plot_embedding(tsne_results, labels, domain_labels, 'training_mode', name)
	# plt.savefig(name+'.pdf')

def plot_embedding(X, y, d, training_mode, save_name):
	x_min, x_max = np.min(X, 0), np.max(X, 0)
	X = (X - x_min) / (x_max - x_min)
	# y = list(itertools.chain.from_iterable(y))
	# y = np.asarray(y)
	clrs = sns.color_palette("husl", 65)
	plt.figure(figsize=(23, 20))
	source_flag = 0
	target_flag = 0
	for i in range(len(d)):  # X.shape[0] : 1024
		# plot colored number
		if d[i] == 0:
			colors = (0.0, 0.0, 1.0, 1.0)
			shape = 'o'
			legend_label = 'Source features'
			marker_size = 10
			source_flag += 1
		else:
			colors = (1.0, 0.0, 0.0, 1.0)
			shape = 'd'
			legend_label = 'Target features'
			marker_size = 10
			target_flag += 1
		# plt.text(X[i, 0], X[i, 1], shape,
		# plt.text(X[i, 0], X[i, 1], y[i],
		# 		color=clrs[y[i]],
		# 		fontdict={'weight': 'bold', 'size': 9})
		if source_flag ==1 or target_flag ==1:
			plt.plot(X[i, 0], X[i, 1], shape, color=clrs[y[i]], markersize=marker_size, mfc='none', label = legend_label)
		else:
			plt.plot(X[i, 0], X[i, 1], shape, color=clrs[y[i]], markersize=marker_size, mfc='none')
	# len_source = int(len(X)/2)
	# plt.plot(X[0:len_source, 0], X[0:len_source, 1], 'o', color=(0.0, 0.0, 1.0, 1.0), markersize=5, label='Source features')
	# plt.plot(X[len_source:, 0], X[len_source:, 1], 'd', color=(1.0, 0.0, 0.0, 1.0), markersize=4, label='Target features')
	plt.legend(loc="lower right", fontsize=50)
	plt.tight_layout()
	plt.xticks([]), plt.yticks([])

	save_folder = 'saved_plot'
	if not os.path.exists(save_folder):
		os.makedirs(save_folder)

	fig_name = 'saved_plot/' + str(training_mode) + '_' + str(save_name) + '.pdf'
	plt.savefig(fig_name)
	print('{} is saved'.format(fig_name))

draw_tsne(feature_s, feature_t, label_s, label_t, name = 'TSL3_AR_tsne')
# draw_tsne(feature_s, feature_t, label_s, label_t, name = 'TSL_AR_tsne')
# draw_tsne(feature_s_sourceonly, feature_t_sourceonly, label_s_sourceonly, label_s_sourceonly, name = 'sourceonly_tsne')

