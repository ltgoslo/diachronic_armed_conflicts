#!/projects/ltg/python3/bin/python3
# coding: utf-8

from helpers import *
from argparse import ArgumentParser
import json
import numpy as np
from os import path

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--testfile', required=True, action='store')
    parser.add_argument('--visual', action='store', type=bool, default=False)
    parser.add_argument('--year', action='store', type=int, default=2010)
    parser.add_argument('--finyear', action='store', type=int, default=2018)
    parser.add_argument('--lmbd', action='store', type=float, default=0.0)
    parser.add_argument('--candidates', action='store', type=int, default=2)
    parser.add_argument('--threshold', action='store_true')
    parser.add_argument('--modeldir', action='store', default='NoW/')
    args = parser.parse_args()

    referencefile = args.testfile
    modeldir = args.modeldir
    data = json.loads(open(referencefile).read())

    all_precisions = []
    all_recalls = []
    all_fscores = []
    all_new_pairs = []

    print('Year\tPrecision\tRecall\tF1\tNew_pairs')

    for cur_year in range(args.year, args.finyear):
        modelfile = path.join(modeldir, '%s_incremental.model' % str(cur_year))
        print('Current model:', modelfile, file=sys.stderr)
        model = load_embeddings(modelfile)
        cur_data = data[str(cur_year)]

        locvecs = []
        insvecs = []

        wars = {}

        for loc in cur_data:
            if cur_data[loc]:
                # print('War in', loc, file=sys.stderr)
                wars[loc] = []
                for el in cur_data[loc]:
                    if el in model:
                        wars[loc].append(el)
                        locvec = model[loc]
                        insvec = model[el]
                        # print(el, file=sys.stderr)
                        locvecs.append(locvec)
                        insvecs.append(insvec)
                    else:
                        print(el, 'not found!', file=sys.stderr)

        print('Whole train dataset shape:', len(locvecs), file=sys.stderr)
        transforms = learn_projection((locvecs, insvecs), model, lmbd=args.lmbd, from_df=False)

        print('Tranformation matrix created', file=sys.stderr)
        sim_average = None
        sim_std = None
        threshold = None
        if args.threshold:
            original_sims = []
            for loc in wars:
                cur_sims = estimate_sims(loc, wars[loc], transforms, model)
                original_sims += cur_sims

            sim_average = np.average(original_sims)
            sim_std = np.std(original_sims)
            print('Average insurgent similarity to projection: %.3f' % sim_average, file=sys.stderr)
            print('Max insurgent similarity: %.3f' % np.max(original_sims), file=sys.stderr)
            print('Min insurgent similarity %.3f' % np.min(original_sims), file=sys.stderr)
            print('Standard deviation of insurgent similarities: %.3f' % sim_std, file=sys.stderr)

        print('Testing on the next year with %d candidates' % args.candidates, file=sys.stderr)

        next_year = cur_year + 1
        nextmodelfile = path.join(modeldir, '%s_incremental.model' % str(next_year))

        print('Test model:', nextmodelfile, file=sys.stderr)
        nextmodel = load_embeddings(nextmodelfile)
        cur_data = data[str(next_year)]

        next_conflicts = 0
        existing_conflicts = 0

        tps = 0  # true positives
        fps = 0  # false positives
        fns = 0  # false negatives

        for loc in cur_data:
            if cur_data[loc]:
                for el in cur_data[loc]:
                    next_conflicts += 1
                    if loc in wars:
                        if el in wars[loc]:
                            existing_conflicts += 1
        new = (next_conflicts - existing_conflicts) / next_conflicts
        print('Ratio of new conflicts in year %d: %.2f' % (next_year, new), file=sys.stderr)
        all_new_pairs.append(new)

        for loc in cur_data:
            candidates, predicted_vector = predict(loc, nextmodel, transforms, topn=args.candidates)

            # Filtering stage
            # We allow only candidates which are not further from the projection
            # than one sigma from the average similarity in the true set
            if args.threshold:
                threshold = sim_average - sim_std
                rejected = [c for c in candidates if c[1] < threshold]
                candidates = [c for c in candidates if c[1] >= threshold]
            else:
                rejected = []
            # End filtering stage

            candidates = [i[0] for i in candidates]

            insurgents = cur_data[loc]

            true_ins = set([i.split('_')[0].lower() for i in insurgents])
            pred_ins = set([i.split('_')[0].lower() for i in candidates])
            if args.visual:
                dots2plot = len(candidates) + len(insurgents) + len(rejected) + 2
                matrix = np.zeros((dots2plot, model.vector_size))
                words = []
                classes = []
                matrix[0, :] = nextmodel[loc]
                words.append(loc.split('_')[0])
                classes.append('Location')

                matrix[1, :] = predicted_vector
                words.append('')
                classes.append('Projection')

                counter = 2
                for word in candidates:
                    matrix[counter, :] = nextmodel[word]
                    words.append(word.split('_')[0])
                    classes.append('Predicted insurgents')
                    counter += 1

                for word in insurgents:
                    matrix[counter, :] = nextmodel[word]
                    words.append(word.split('_')[0])
                    classes.append('True insurgents')
                    counter += 1

                for word in rejected:
                    matrix[counter, :] = nextmodel[word[0]]
                    words.append(word[0].split('_')[0])
                    classes.append('Rejected insurgents')
                    counter += 1

                visualize(
                    words, matrix, classes, fname=str(next_year) + '_' + loc.split('_')[0].lower(),
                    radius=1-threshold)

            true_positives = len(true_ins & pred_ins)
            false_positives = len(pred_ins - true_ins)
            false_negatives = len(true_ins - pred_ins)

            tps += true_positives
            fps += false_positives
            fns += false_negatives

            # print('True:', loc, true_ins, file=sys.stderr)
            # print('Predicted:', loc, pred_ins, file=sys.stderr)
            # print('Rejected:', rejected, file=sys.stderr)
            # print('%d TPs, %d FPs, %d FNs'
            #      % (true_positives, false_positives, false_negatives), file=sys.stderr)

        try:
            cur_precision = tps / (tps + fps)
        except ZeroDivisionError:
            cur_precision = 0
        cur_recall = tps / (tps + fns)
        try:
            cur_f1 = 2 * (cur_precision * cur_recall) / (cur_precision + cur_recall)
        except ZeroDivisionError:
            cur_f1 = 0
        print('%d\t%.2f\t%.2f\t%.2f\t%.2f' % (next_year, cur_precision, cur_recall, cur_f1, new))
        all_precisions.append(cur_precision)
        all_recalls.append(cur_recall)
        all_fscores.append(cur_f1)

        print('Average F1 score: %.3f' % cur_f1, file=sys.stderr)

    print('Average\t%.3f\t%.3f\t%.3f\t%.2f'
          % (np.average(all_precisions), np.average(all_recalls), np.average(all_fscores),
             np.average(all_new_pairs)))
