#! python3
# coding: utf-8
from helpers import *
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--modelfile', required=True, action='store')
    parser.add_argument('--reference', required=True, action='store')
    parser.add_argument('--skip', action='store', type=bool, default=False)
    parser.add_argument('--lmbd', action='store', type=float, default=0.0)
    args = parser.parse_args()

    modelfile = args.modelfile
    referencefile = args.reference

    currentyear = referencefile.split('/')[1].split('_')[0]
    currentyear = int(currentyear)
    print('Current training year:', currentyear, file=sys.stderr)

    skip = args.skip

    model = load_embeddings(modelfile)

    df, _ = load_dataset(referencefile, embedding=model)

    train_pairs = list(zip(df['Location'], df['Insurgent']))

    df['LocVec'] = df['Location'].apply(get_vector, emb=model)
    df['InsVec'] = df['Insurgent'].apply(get_vector, emb=model)
    print('Whole dataset shape:', df.shape, file=sys.stderr)

    transforms = learn_projection(df, model, lmbd=args.lmbd)

    print('Tranformation matrix created', file=sys.stderr)
    # print(transforms.shape, file=sys.stderr)
    print('Testing on the next years', file=sys.stderr)

    # print('Year\tAccuracy@1\tAccuracy@5\tAccuracy@10\tAccuracy@1_new\tAccuracy@5_new\t'
    #      'Accuracy@10_new\tOOV\tNew pairs')
    next_year = currentyear + 1

    print('Now testing on year:', next_year, file=sys.stderr)
    testmodelfile = modelfile.replace(str(currentyear), str(next_year))
    testmodel = load_embeddings(testmodelfile)

    testfile = referencefile.replace(str(currentyear), str(next_year))

    df, test_unknown = load_dataset(testfile, embedding=testmodel, evaluation=True)
    size = df.shape[0]

    print('Whole test dataset shape:', df.shape, file=sys.stderr)

    accuracies1 = []
    accuracies5 = []
    accuracies10 = []

    for loc, ins in zip(df['Location'], df['Insurgent']):
        candidates = predict(loc, testmodel, transforms)
        # print >> sys.stderr, candidates
        accuracy1, accuracy5, accuracy10 = calc_accuracies(candidates, ins)
        accuracies1.append(accuracy1)
        accuracies5.append(accuracy5)
        accuracies10.append(accuracy10)

    # Now goes the unknown:

    accuracies1_new = []
    accuracies5_new = []
    accuracies10_new = []
    print('Predictions for the unknown pairs:', file=sys.stderr)
    for loc, ins in zip(df['Location'], df['Insurgent']):
        if (loc, ins) in train_pairs:
            # Was in the training set
            continue
        candidates = predict(loc, testmodel, transforms)
        print((loc, ins), candidates[:5], file=sys.stderr)
        accuracy1_new, accuracy5_new, accuracy10_new = calc_accuracies(candidates, ins)
        # print('Accuracy @1:', accuracy1_new, file=sys.stderr)
        # print('Accuracy @5:', accuracy5_new, file=sys.stderr)
        # print('Accuracy @10:', accuracy10_new, file=sys.stderr)
        accuracies1_new.append(accuracy1_new)
        accuracies5_new.append(accuracy5_new)
        accuracies10_new.append(accuracy10_new)

    print(next_year, '\t', np.average(accuracies1), '\t', np.average(accuracies5), '\t',
          np.average(accuracies10), '\t', np.average(accuracies1_new), '\t',
          np.average(accuracies5_new), '\t', np.average(accuracies10_new), '\t',
          test_unknown / size, '\t', len(accuracies1_new) / len(accuracies1))
