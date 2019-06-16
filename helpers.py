#!/usr/bin/env python3
# coding: utf-8

import sys
import numpy as np
import matplotlib  # For use on servers
matplotlib.use('Agg')
from gensim import models
import pandas as pd
import logging
import requests
import json
from sklearn.decomposition import PCA
import pylab as plot
from gensim.matutils import unitvec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def learn_projection(dataset, embedding, lmbd=1.0, save2file=None, from_df=True):
    if from_df:
        locvectors = dataset['LocVec'].T
        insvectors = dataset['InsVec'].T
    else:
        locvectors = dataset[0]
        insvectors = dataset[1]
    locvectors = np.mat([[i for i in vec] for vec in locvectors])
    insvectors = np.mat([[i for i in vec] for vec in insvectors])
    m = len(locvectors)
    x = np.c_[np.ones(m), locvectors]  # Adding bias term to the source vectors

    num_features = embedding.vector_size

    # Build initial zero transformation matrix
    learned_projection = np.zeros((num_features, x.shape[1]))
    learned_projection = np.mat(learned_projection)

    for component in range(0, num_features):  # Iterate over input components
        y = insvectors[:, component]  # True answers
        # Computing optimal transformation vector for the current component
        cur_projection = normalequation(x, y, lmbd, num_features)

        # Adding the computed vector to the transformation matrix
        learned_projection[component, :] = cur_projection.T

    if save2file:
        # Saving matrix to file:
        np.savetxt(save2file, learned_projection, delimiter=',')
    return learned_projection


def normalequation(data, target, lambda_value, vector_size):
    regularizer = 0
    if lambda_value != 0:  # Regularization term
        regularizer = np.eye(vector_size + 1)
        regularizer[0, 0] = 0
        regularizer = np.mat(regularizer)
    # Normal equation:
    theta = np.linalg.pinv(data.T * data + lambda_value * regularizer) * data.T * target
    return theta


def load_embeddings(modelfile):
    if modelfile.endswith('.txt.gz') or modelfile.endswith('.txt'):
        model = models.Word2Vec.load_word2vec_format(modelfile, binary=False)
    elif modelfile.endswith('.bin.gz') or modelfile.endswith('.bin'):
        model = models.Word2Vec.load_word2vec_format(modelfile, binary=True)
    else:
        # model = models.Word2Vec.load(modelfile)
        model = models.KeyedVectors.load(modelfile)  # For newer models
    model.init_sims(replace=True)
    return model


def get_vector(word, emb=None):
    if not emb:
        return None
    vector = emb[word]
    return vector


def load_dataset(datafile, embedding, evaluation=False):
    df = pd.read_csv(datafile, sep='\t', header=0)
    pairs = list(zip(df['Location'], df['Insurgent']))
    test_unknown = 0
    not_found = set()
    for loc, ins in pairs:
        if loc not in embedding:
            not_found.add(loc)
            if evaluation:
                test_unknown += 1
            else:
                df = df[df.Location != loc]
        if ins not in embedding:
            not_found.add(ins)
            if evaluation:
                test_unknown += 1
            else:
                df = df[df.Insurgent != ins]
    if not_found:
        print('Not found in the model:', sorted(not_found), file=sys.stderr)
    return df, test_unknown


def estimate_sims(location, insurgents, projection, model):
    #  Finding how far away are true insurgents from transformed locations
    test = np.mat(model[location])
    test = np.c_[1.0, test]  # Adding bias term
    predicted_vector = np.dot(projection, test.T)
    predicted_vector = np.squeeze(np.asarray(predicted_vector))
    insvecs = [model[insurgent] for insurgent in insurgents]
    sims = [np.dot(unitvec(predicted_vector), unitvec(insvec)) for insvec in insvecs]
    return sims


def predict(location, embedding, projection, topn=10):
    test = np.mat(embedding[location])
    test = np.c_[1.0, test]  # Adding bias term
    predicted_vector = np.dot(projection, test.T)
    predicted_vector = np.squeeze(np.asarray(predicted_vector))
    # Our predictions:
    nearest_neighbors = embedding.most_similar(positive=[predicted_vector], topn=topn)
    return nearest_neighbors, predicted_vector


def calc_accuracies(candidates, insurgent):
    accuracy1 = 0
    accuracy5 = 0
    accuracy10 = 0
    correct = insurgent.split('_')[0].lower()
    if correct in candidates:
        accuracy10 = 1
        if correct in candidates[:5]:
            accuracy5 = 1
            if candidates[0] == correct:
                accuracy1 = 1
    return accuracy1, accuracy5, accuracy10


def tag_ud(port, text='Do not forget to pass some text as a string!'):
    # UDPipe tagging for any language you have a model for.
    # Demands UDPipe REST server (https://ufal.mff.cuni.cz/udpipe/users-manual#udpipe_server)
    # running on a port defined in webvectors.cfg
    # Start the server with something like:
    # udpipe_server --daemon 66666 MyModel MyModel /opt/my.model UD

    # Sending user query to the server:
    ud_reply = requests.post('http://localhost:%s/process' % port,
                             data={'tokenizer': '', 'tagger': '', 'data': text}).content

    # Getting the result in the CONLLU format:
    processed = json.loads(ud_reply.decode('utf-8'))['result']

    # Skipping technical lines:
    content = [l for l in processed.split('\n') if not l.startswith('#')]

    # Extracting lemmas and tags from the processed queries:
    tagged = [w.split('\t')[2] + '_' + w.split('\t')[3] for w in content if w]

    return tagged


def jaccard(y, pred):
    intersection = y.intersection(pred)
    union = y | pred
    score = len(intersection) / len(union)
    return score


def visualize(words, matrix, classes, fname=False, radius=None):
    embedding = PCA(n_components=2)
    y = embedding.fit_transform(matrix)

    colors = {'Location': 'brown', 'Projection': 'blue', 'Predicted insurgents': 'red',
              'True insurgents': 'green', 'Rejected insurgents': 'black'}
    shapes = {'Location': '*', 'Projection': 'x', 'Predicted insurgents': 'o',
              'True insurgents': 'o', 'Rejected insurgents': 'o'}

    class2color = [colors[w] for w in classes]

    class2shape = [shapes[w] for w in classes]

    xpositions = y[:, 0]
    ypositions = y[:, 1]
    seen = set()

    plot.clf()
    circle = None

    if radius:
        circle = plot.Circle((xpositions[1], ypositions[1]), radius=radius, color='r',
                             fill=False, label='Hypersphere')
        plot.gcf().gca().add_artist(circle)

    for word, class_label, x, y, color, shape in \
            zip(words, classes, xpositions, ypositions, class2color, class2shape):
        plot.scatter(x, y, 40, alpha=0.8, marker=shape, color=color,
                     label=class_label if class_label not in seen else "")
        seen.add(class_label)

        lemma = word.replace('::', ' ')
        plot.annotate(lemma, xy=(x, y),
                      size='large' if class_label == 'Rejected insurgents' else 'x-large',
                      weight='bold', color=color)

    plot.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plot.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    main_legend = plot.legend(loc='best')
    if radius:
        hsphere_legend = plot.legend(loc=4, handles=[circle])
        plot.gca().add_artist(hsphere_legend)
        plot.gca().add_artist(main_legend)

    if fname:
        fname = fname + '.png'
        plot.savefig(fname, dpi=150, bbox_inches='tight')
    else:
        plot.show()
    plot.close()
    plot.clf()
