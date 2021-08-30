
import argparse
import json
import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from tqdm import tqdm


clip_model_types = ['clip-single_cls-maxpool',
                    'clip-single_cls-meanpool',
                    'clip-single_cls-random_index',
                    'clip-single_cls-two_random_index',
                    'clip-zero_shot_cls-maxpool',
                    'clip-zero_shot_cls-meanpool',
                    'clip-zero_shot_cls-random_index',
                    'clip-zero_shot_cls-two_random_index']
rotator_model_types = ['clip-rotator-two_random_index']

THRESH = 0.05
welchs_opts = {'equal_var': False,
               'alternative': 'two-sided'}


def main(args):
    # Assemble validation and test results.
    d = []
    erroneous_result_files = []
    missing_result_files = []
    for model_types, prefix_dir, aux in \
            [[clip_model_types, "%s_seed_" % args.clip_results_dir_prefix, 'none'],
             [rotator_model_types, "%s_init_seed_" % args.rotator_results_dir_prefix, 'init'],
             [rotator_model_types, "%s_final_seed_" % args.rotator_results_dir_prefix, 'final'],
             [rotator_model_types, "%s_init_final_seed_" % args.rotator_results_dir_prefix, 'both']]:
        for model in model_types:
            n_seeds = 1 if 'zero_shot' in model else args.n_seeds  # zero-shot models have no inference-time variance
            for seed in range(n_seeds):

                # Read in validation results.
                fn = os.path.join("%s%d" % (prefix_dir, seed), model, 'vl-results-%d.json' % seed)
                if not os.path.isfile(fn):
                    missing_result_files.append(fn)
                    continue
                with open(fn, 'r') as f:
                    seed_results = json.load(f)

                    # Check that the result isn't from a pytorch lightning sanity check.
                    if seed_results['vl/all_view_res']['0']['val_total'] < 2000:
                        erroneous_result_files.append(fn)

                    entry = {'model': model,
                             'aux': aux,
                             'seed': seed,
                             'fold': 'val',
                             'acc': seed_results['vl/acc'],
                             'acc_nonvis': seed_results['vl/acc_nonvis'],
                             'acc_visual': seed_results['vl/acc_visual']}
                    d.append(entry)

                # Compute test results.
                if args.test_set_answers_fn:
                    fn = os.path.join("%s%d" % (prefix_dir, seed), model, 'test-results-%d.json' % seed)
                    if not os.path.isfile(fn) or args.force_calculate_test_results:  # calculate test results
                        model_family = 'zeroshotclassifier'
                        if 'single_cls' in model:
                            model_family = 'singleclassifier'
                        if 'rotator' in model:
                            model_family = 'rotator'
                        results_fn = os.path.join("%s%d" % (prefix_dir, seed), model, '%s_test_results.json' % model_family)
                        if not os.path.isfile(results_fn):
                            missing_result_files.append(results_fn)
                            continue
                        seed_results = compute_test_metrics(args.test_set_answers_fn, results_fn)
                        # Write test results so we don't have to do this again.
                        with open(fn, 'w') as f:
                            json.dump(seed_results, f)
                    else:
                        with open(fn, 'r') as f:
                            seed_results = json.load(f)

                    entry = {'model': model,
                             'aux': aux,
                             'seed': seed,
                             'fold': 'test',
                             'acc': seed_results['test/acc'],
                             'acc_nonvis': seed_results['test/acc_nonvis'],
                             'acc_visual': seed_results['test/acc_visual']}
                    d.append(entry)

    # Data statistics and tests.
    df = pd.DataFrame(d)

    comparisons = [(('clip-single_cls-maxpool', 'none'), ('clip-single_cls-meanpool', 'none')),
                   (('clip-rotator-two_random_index', 'both'), ('clip-single_cls-maxpool', 'none')),
                   (('clip-rotator-two_random_index', 'both'), ('clip-single_cls-two_random_index', 'none')),
                   ]
    comp_folds = ['val', 'test']
    comp_metrics = ['acc']  # , 'acc_nonvis', 'acc_visual']
    for fold in comp_folds:
        print('fold=%s' % fold)
        for (model_a, aux_a), (model_b, aux_b) in comparisons:
            print("(%s, %s) compared to (%s, %s)" % (model_a, aux_a, model_b, aux_b))
            for metric in comp_metrics:
                a = df.loc[(df['model'] == model_a) & (df['aux'] == aux_a) & (df['fold'] == fold)][metric]
                b = df.loc[(df['model'] == model_b) & (df['aux'] == aux_b) & (df['fold'] == fold)][metric]
                print('\t%s\t\t\t\t\t\tmean\tstd\tN' % metric)
                print('\t\t%s\t%.3f\t%.3f\t%d' % (model_a, np.mean(a), np.std(a), len(a)))
                print('\t\t%s\t%.3f\t%.3f\t%d' % (model_b, np.mean(b), np.std(b), len(b)))
                t, p = ttest_ind(a, b, **welchs_opts)
                print('\t\t\tp=%f; sig=%d' %
                      (p, 1 if p < THRESH / ((len(comp_folds) * len(comparisons) * len(comp_metrics)) - 1) else 0))
                # Subtract one from Bonferroni correction because we don't actually want to run/care about
                # the maxpool/meanpool comparison on the test fold.

    # Populate LaTeX table
    #     [model] & [views] & [v viz] & [v nonviz] & [v all] & [t viz] & [t nonviz] & [t all]
    #     CLIP & 360 & 84.5 & 66.1 & 75.3 & 80.0 & 61.4 & 70.9 \\
    #     \scorer & 360 & \bf 90.6 & \bf 79.3 &  \bf 85.0 & \bf 85.9 & 71.3 & \bf 78.7 \\
    #     \midrule
    #     CLIP & Single & 79.5 & 65.2 & 72.3 & 73.9 & 60.4 & 67.3 \\
    #     \scorer\ & Single & \bf 89.4 & \bf 75.6 & \bf 82.5 & \bf 84.1 & \bf 69.6 & \bf 77.0 \\
    #     \midrule
    #     CLIP & Two & 81.7 & 65.5 & 73.6 & 76.2 & 61.0 & 68.8 \\
    #     \scorer\ & Two & 91.2 & 75.1 & 83.2 & 85.8 & 70.9 & 78.5 \\
    #     \model\ & Two & \B{91.5} & \B{81.2} & \B{86.3} & \B{86.6} & \B{72.0} & \B{79.4} \\
    for comp_set in  \
            [[['clip-zero_shot_cls-maxpool', 'CLIP', '360-max', 'none'],
             ['clip-zero_shot_cls-meanpool', 'CLIP', '360-mean', 'none'],
             ['clip-single_cls-maxpool', '\\scorer', '360-max', 'none'],
             ['clip-single_cls-meanpool', '\\scorer', '360-mean', 'none']],
             [['clip-zero_shot_cls-random_index', 'CLIP', 'Single', 'none'],
             ['clip-single_cls-random_index', '\\scorer', 'Single', 'none']],
             [['clip-zero_shot_cls-two_random_index', 'CLIP', 'Two', 'none'],
             ['clip-single_cls-two_random_index', '\\scorer', 'Two', 'none'],
             ['clip-rotator-two_random_index', '\\model-init', 'Two', 'init'],
             ['clip-rotator-two_random_index', '\\model-final', 'Two', 'final'],
             ['clip-rotator-two_random_index', '\\model-both', 'Two', 'both']],
             ]:
        for model, model_str, views, aux in comp_set:
            ss = ['%s & %s' % (model_str, views)]
            for fold in ['val', 'test']:
                for metric in ['acc_visual', 'acc_nonvis', 'acc']:
                    a = df.loc[(df['model'] == model) & (df['fold'] == fold) & (df['aux'] == aux)][metric]
                    ss.append('%.1f (%.1f)' % (np.mean(a) * 100., np.std(a) * 100.))
            print(' & '.join(ss) + ' \\\\')
        print('\\midrule')

    if len(missing_result_files) > 0:
        print('WARNING: The following results files are expected but were not found; results may shift')
        print('\n'.join(missing_result_files))

    if len(erroneous_result_files) > 0:
        print('WARNING: The following results files are likely bad perf estimates from PTL sanity checks')
        print('\n'.join(erroneous_result_files))


# answers_fn - filepath to answers_json
# output_fn - filepath to output dump, e.g., zeroshotclassifier_test_results.json
def compute_test_metrics(answers_fn, output_fn):
    # load JSONs
    with open(answers_fn, 'r') as f:
        answers = json.load(f)

    with open(output_fn, 'r') as f:
        output = json.load(f)

    num_views = 8
    n_view_res = {}
    mode = 'test'

    for view in range(num_views):
        print(f"processing view: {view}")

        view_res = {
            'correct': 0,
            'pl_correct': 0,
            'total': 0,

            'visual_correct': 0,
            'pl_visual_correct': 0,
            'visual_total': 0,

            'nonvis_correct': 0,
            'pl_nonvis_correct': 0,
            'nonvis_total': 0,
        }

        for idx, o in enumerate(tqdm(output[str(view)])):
            # pdb.set_trace()

            assert (o['objects'] == answers[idx]['objects']), \
                'Prediction instance does not match answers ' + str(o['objects']) + ' ' + str(answers[idx]['objects'])
            pred_ans = o['pred_ans']
            corr_ans = answers[idx]['ans']
            correct = (pred_ans == corr_ans)

            num_steps = o['num_steps']
            is_visual = answers[idx]['visual']

            if correct:
                view_res['correct'] += 1
                view_res['pl_correct'] += 1. / num_steps
            view_res['total'] += 1

            if is_visual:
                if correct:
                    view_res['visual_correct'] += 1
                    view_res['pl_visual_correct'] += 1. / float(num_steps)
                view_res['visual_total'] += 1
            else:
                if correct:
                    view_res['nonvis_correct'] += 1
                    view_res['pl_nonvis_correct'] += 1. / float(num_steps)
                view_res['nonvis_total'] += 1

        view_res['acc'] = float(view_res['correct']) / view_res['total']
        view_res['pl_acc'] = float(view_res['pl_correct']) / view_res['total']

        view_res['visual_acc'] = float(view_res['visual_correct']) / view_res['visual_total']
        view_res['pl_visual_acc'] = float(view_res['pl_visual_correct']) / view_res['visual_total']

        view_res['nonvis_acc'] = float(view_res['nonvis_correct']) / view_res['nonvis_total']
        view_res['pl_nonvis_acc'] = float(view_res['pl_nonvis_correct']) / view_res['nonvis_total']

        n_view_res[view] = view_res

    acc = sum([r['correct'] for r in n_view_res.values()]) / float(sum([r['total'] for r in n_view_res.values()]))
    visual_acc = sum([r['visual_correct'] for r in n_view_res.values()]) / float(
        sum([r['visual_total'] for r in n_view_res.values()]))
    nonvis_acc = sum([r['nonvis_correct'] for r in n_view_res.values()]) / float(
        sum([r['nonvis_total'] for r in n_view_res.values()]))

    pl_acc = sum([r['pl_correct'] for r in n_view_res.values()]) / float(sum([r['total'] for r in n_view_res.values()]))
    pl_visual_acc = sum([r['pl_visual_correct'] for r in n_view_res.values()]) / float(
        sum([r['visual_total'] for r in n_view_res.values()]))
    pl_nonvis_acc = sum([r['pl_nonvis_correct'] for r in n_view_res.values()]) / float(
        sum([r['nonvis_total'] for r in n_view_res.values()]))

    res = {
        f'{mode}/acc': acc,
        f'{mode}/acc_visual': visual_acc,
        f'{mode}/acc_nonvis': nonvis_acc,
        f'{mode}/pl_acc': pl_acc,
        f'{mode}/pl_acc_visual': pl_visual_acc,
        f'{mode}/pl_acc_nonvis': pl_nonvis_acc,
        f'{mode}/all_view_res': n_view_res,
    }

    # results to save
    results_dict = dict(res)

    best_acc = results_dict[f'{mode}/acc']
    best_acc_visual = results_dict[f'{mode}/acc_visual']
    best_acc_nonvis = results_dict[f'{mode}/acc_nonvis']
    best_pl_acc = results_dict[f'{mode}/pl_acc']
    best_pl_acc_visual = results_dict[f'{mode}/pl_acc_visual']
    best_pl_acc_nonvis = results_dict[f'{mode}/pl_acc_nonvis']

    # print best result
    print("\nBest-----:")
    print(
        f'Best {mode} Acc: {best_acc:0.5f} ({best_pl_acc:0.5f}) | Visual {best_acc_visual:0.5f} ({best_pl_acc_visual:0.5f}) | Nonvis: {best_acc_nonvis:0.5f} ({best_pl_acc_nonvis:0.5f}) ')
    print("------------")
    return results_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_results_dir_prefix', type=str, required=True,
                        help='CLIP and MATCH results dir prefix before adding seed')
    parser.add_argument('--rotator_results_dir_prefix', type=str, required=True,
                        help='Rotator results dir prefix before adding seed and losses')
    parser.add_argument('--n_seeds', type=int, required=True,
                        help='The number of seeds to index')
    parser.add_argument('--test_set_answers_fn', type=str, required=False,
                        help='The test set annotations for final test eval; not publicly available')
    parser.add_argument('--force_calculate_test_results', action='store_true')
    args = parser.parse_args()
    main(args)
