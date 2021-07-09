#import tensorflow as tf
import argparse
import logging
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import pairwise_distances
import scipy
import hdbscan
import pandas as pd
from time import perf_counter

parser = argparse.ArgumentParser(description='Apply cluster method to similarity matrix.')
parser.add_argument('--method', dest='method', action='store', type=str,
                    default='HACsingle',
                    choices=['HIST','PLOT','HACsingle','HACcomplete','HDBSCAN','PERCEPTRON'],
                    help='Optional, method')
parser.add_argument('--validation-file', dest='validationfile', action='store', type=str,
                    help='filepath to CSV file containing correct clustering in Group column.')
parser.add_argument('--similarity-file', dest='simmatrixfile', action='store', type=str, default='../node/output/SimilarityMatrix_ING_DiceSq.bin',
                    help='filepath to binary file containing similarity matrix (float32).')
parser.add_argument('--out-name', dest='outname', action='store', type=str, default='ING_DiceSq',
                    help='name to append to outfiles.')
parser.add_argument('--latex', dest='latex', action='store', type=bool, default=False,
                    help='Latex friendly markup')

args = parser.parse_args()


def formula(x):
    i, v = x
    if(v<0):
        return v-i
    return v+1

def fix_labels_noise_and_zero(input_labels):
    indexed = enumerate(input_labels)
    res = np.fromiter(map(formula, indexed), dtype=int)
    return res

#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

filename = args.simmatrixfile #'../node/output/SimilarityMatrix_ING_DiceSq.bin'
with open(filename, 'rb') as f:
    simmatrix = np.fromfile(f, dtype=np.float32)
if(args.latex==False):
	print(f'Read file "{filename}" containing a distance matrix of shape: {simmatrix.size}')
distmatrix = np.reshape(np.subtract(1.0,simmatrix), [math.isqrt(simmatrix.size), math.isqrt(simmatrix.size)])
np.fill_diagonal(distmatrix, 0.0)
distmatrix_condensed = scipy.spatial.distance.squareform(distmatrix)

validation_filename = args.validationfile #'../node/data/example2_puntcomma_delimited.csv'
groups = None
if validation_filename is not None:
    validation_df = pd.read_csv(validation_filename, sep=';',quotechar='"')
    groups = validation_df.Group

def print_scores(_distmatrix,_labels,_truth=None):
    rand = None
    rand_adjusted = None
    if _truth is not None:
        rand = metrics.rand_score(_truth, _labels)
        rand_adjusted = metrics.adjusted_rand_score(_truth, _labels)
        if(args.latex==False):
            print(f'rand score: {rand}')
            print(f'adjusted rand score: {rand_adjusted}')

    calinski = metrics.calinski_harabasz_score(_distmatrix, _labels)
    silhouette = metrics.silhouette_score(_distmatrix, _labels, metric='euclidean')
    if(args.latex==False):
        print(f'calinksi harabasz score: {calinski}')
        print(f'silhouette_score score: {silhouette}')
    elif(rand==None):
        print(f'- & - & {calinski:.1} & {silhouette:.4}')
    else:
        print(f'{rand:.4f} & {rand_adjusted:.3f} & {calinski:.1f} & {silhouette:.4f}')
if(args.method == 'HIST'):
    plt.hist(simmatrix, bins=300, histtype='stepfilled',log=True)
    plt.ylim(1.0,500000.0)
    plt.xlim(0.0,1.0)
    plt.show()
elif(args.method == 'PLOT'):
    input = np.loadtxt('./output/HACCOMPLETElabels_'+args.outname+'.txt',dtype=int,skiprows=1)
    labels, counts = np.unique(input,return_counts=True)
    y=sorted(counts,reverse=True)
    x=list(range(len(counts)))
    #x,y = np.unique(counts,return_counts=True)
    plt.bar(x,y, linewidth=0.0,log=True)
    plt.show()
elif(args.method == 'HACsingle'):
    Z = scipy.cluster.hierarchy.linkage(distmatrix_condensed, method='single', optimal_ordering=False)
    labels = scipy.cluster.hierarchy.fcluster(Z, 0.3, criterion='distance', depth=2, R=None, monocrit=None)
    if(args.latex==False):
	    print(labels)
    print_scores(distmatrix,labels,groups)

    np.savetxt('./output/HACSINGLElabels_'+args.outname+'.txt',labels.astype(int),fmt='%i',header="Group", comments='')
	#fig = plt.figure(figsize=(25, 10))
	#dn = scipy.cluster.hierarchy.dendrogram(Z)
	#plt.show()
elif(args.method == 'HACcomplete'):
    Z = scipy.cluster.hierarchy.linkage(distmatrix_condensed, method='complete', optimal_ordering=False)
    labels = scipy.cluster.hierarchy.fcluster(Z, 0.5, criterion='distance', depth=2, R=None, monocrit=None)
    if(args.latex==False):
        print(labels)
    print_scores(distmatrix,labels,groups)

    np.savetxt('./output/HACCOMPLETElabels_'+args.outname+'.txt',labels.astype(int),fmt='%i',header="Group", comments='')
	#fig = plt.figure(figsize=(25, 10))
	#dn = scipy.cluster.hierarchy.dendrogram(Z)
	#plt.show()
elif(args.method == 'HDBSCAN'):
    clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True, gen_min_span_tree=True, leaf_size=40, memory='./cache/', metric='precomputed', min_cluster_size=2, min_samples=None, p=None)
    clusterer.fit(distmatrix.astype(np.float64))
	# clusterer.labels_
    np.savetxt('./output/HDBSCANlabels_'+args.outname+'.txt',clusterer.labels_.astype(int),fmt='%i',header="Group", comments='')
	# fix noise and zero
    labels = fix_labels_noise_and_zero(clusterer.labels_)

    print_scores(distmatrix,labels,groups)
elif(args.method == 'PERCEPTRON'):
    labels = np.loadtxt('../node/output/PERCEPTRONlabels_'+args.outname+'.txt',dtype=int,skiprows=1)
    if(args.latex==False):
        print(labels)
    print_scores(distmatrix, labels, groups)


