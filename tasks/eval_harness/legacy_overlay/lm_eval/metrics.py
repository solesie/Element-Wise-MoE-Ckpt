"""Minimal lm-eval 0.2 metrics used by the fixed six-task protocol."""

import math


def mean(items):
    return sum(items) / len(items)


def sample_stddev(items):
    average = mean(items)
    return math.sqrt(sum((item - average) ** 2 for item in items) / (len(items) - 1))


def mean_stderr(items):
    if len(items) < 2:
        return 0.0
    return sample_stddev(items) / math.sqrt(len(items))


def weighted_mean(items):
    values, weights = zip(*items)
    return sum(values) / sum(weights)


def weighted_perplexity(items):
    return math.exp(-weighted_mean(items))


def bits_per_byte(items):
    return -weighted_mean(items) / math.log(2)


def stderr_for_metric(metric, bootstrap_iters):
    del bootstrap_iters
    return mean_stderr if metric is mean else None
