import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import sys
import multiprocessing
import itertools
import functools
from statsmodels.stats.multitest import multipletests

#Let's set the random seed so the results of the notebook
# are always the same at every run
np.random.seed(0)

def generate_dataset(n_datasets, n_null_true, n_samples=100, seed=0):

    # This is to make the results predictable
    np.random.seed(seed)

    n_null_false = n_datasets - n_null_true

    w1 = []
    w2 = []
    null_status = []

    for i in range(n_null_true):

        wn_1 = np.random.normal(loc=90, scale=10, size=n_samples)
        wn_2 = np.random.normal(loc=90, scale=10, size=n_samples)

        w1.append(wn_1)
        w2.append(wn_2)

        null_status.append(True)

    for i in range(n_null_false):

        wn_1 = np.random.normal(loc=95, scale=10, size=n_samples)
        wn_2 = np.random.normal(loc=90, scale=10, size=n_samples)

        w1.append(wn_1)
        w2.append(wn_2)
        null_status.append(False)

    return w1, w2, np.array(null_status)


def worker_function(i, generate_dataset_kw, test, null_hyp_status, alpha):

    generate_dataset_kw['seed'] = (i+1) * 1000

    w1, w2, _ = generate_dataset(**generate_dataset_kw)
    pvalue = test(w1, w2)

    return null_hyp_status(pvalue, alpha)


def measure_rejection_prob(n_iter, test, null_hyp_status,
                           alpha, **generate_dataset_kw):

    n_rejected = 0

    worker = functools.partial(worker_function, 
                generate_dataset_kw=generate_dataset_kw,
                test=test, null_hyp_status=null_hyp_status,
                alpha=alpha)

    pool = multiprocessing.Pool()

    try:

        # for i, res in enumerate(pool.imap(worker, range(n_iter), chunksize=100)):
        for i in range(n_iter):
            res = worker(i)

            if not res:

                n_rejected += 1

            if (i+1) % 100 == 0:

                sys.stderr.write("\r%i out of %i completed (fraction of "
                                 "rejections so far: %.2f)" % (i+1, n_iter,
                                     n_rejected / float(i+1)))
        sys.stderr.write("\n")
        sys.stderr.flush()

    except:

        raise

    finally:

        pool.close()
        pool.join()

    return n_rejected / float(n_iter)


def worker_function2(i, generate_dataset_kw, test, method, alpha):

    generate_dataset_kw['seed'] = (i+1) * 1000

    w1, w2, null_hyp = generate_dataset(**generate_dataset_kw)
    pvalues = test(w1, w2)

    reject, _, _, _ = multipletests(pvalues, alpha,
                                    method=method,
                                    is_sorted=False,
                                    returnsorted=False)

    # False positives: I rejected when I shouldn't have
    n_false_pos = np.sum((reject == True) & (null_hyp == True))

    # False negatives: I didn't reject when I should have
    n_false_neg = np.sum((reject == False) & (null_hyp == False))

    return np.sum(reject), n_false_pos, n_false_neg


def measure_detections(n_iter, test, method,
                       alpha, **generate_dataset_kw):

    n_false_pos = []
    n_false_neg = []
    n_selected = []

    worker = functools.partial(worker_function2, generate_dataset_kw=generate_dataset_kw,
                               test=test, method=method, alpha=alpha)

    pool = multiprocessing.Pool()

    try:
        # for i, (s, fp, fn) in enumerate(pool.imap(worker,
        #                                           range(n_iter),
        #                                           chunksize=100)):
        for i in range(n_iter):
            s, fp, fn = worker(i)
            n_selected.append(s)
            n_false_pos.append(fp)
            n_false_neg.append(fn)

    except:

        raise

    finally:

        pool.close()
        pool.join()

    global_typeI = np.sum(np.array(n_false_pos) > 0) / float(n_iter)

    return (np.average(n_selected),
            np.average(n_false_pos),
            np.average(n_false_neg),
            global_typeI)


def characterize_methods(test, methods, ms, n_false, niter=800, plot=True):

    selections = {}
    false_positives = {}
    false_negatives = {}
    global_typeI = {}

    for method, alpha in methods:

        # Clear output
        sys.stderr.write("Method %s with alpha %.2f" % (method, alpha))

        s = np.zeros(len(ms), int)
        fp = np.zeros_like(s)
        fn = np.zeros_like(s)
        gtI = np.zeros(s.shape[0], float)

        for i, (m, nf) in enumerate(zip(ms, n_false)):

            s[i], fp[i], fn[i], gtI[i] = measure_detections(niter,
                                                            test,
                                                            method,
                                                            alpha,
                                                            n_datasets=m,
                                                            n_null_true=m - nf)
            sys.stderr.write(".")

        selections[(method, alpha)] = s
        false_positives[(method, alpha)] = fp
        false_negatives[(method, alpha)] = fn
        global_typeI[(method, alpha)] = gtI

        sys.stderr.write("completed\n")

    if plot:

        fig, subs = plt.subplots(3, 1, sharex=True,
                                 figsize=(4, 10),
                                 gridspec_kw={'hspace': 0.0, 'top': 0.95})

        for key in methods:

            true_positives = selections[key] - false_positives[key]

            precision = true_positives.astype(float) / selections[key]
            recall = true_positives / \
                (true_positives + false_negatives[key]).astype(float)

            label = r"%s ($\alpha$=%.2f)" % (key[0], key[1])

            _ = subs[0].plot(ms, precision, label=label)
            _ = subs[1].plot(ms, recall, label=label)
            _ = subs[2].plot(ms, global_typeI[key], label=label)

        subs[0].set_ylabel("Precision\n(purity)")
        subs[1].set_ylabel("Recall\n(completeness)")
        subs[2].set_ylabel(r"Global $\alpha$")
        subs[2].set_xlabel("Number of tests")

        subs[2].set_xscale("log")

        plt.axes(subs[0])
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    return selections, false_positives, false_negatives, global_typeI
