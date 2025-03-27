import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist

trackers = []
dataset_name = 'dtb70'
dataset_name = 'uavdt'
dataset_name = 'visdrone2018'
# dataset_name = 'uav123'

"""ortrack"""
trackers.extend(trackerlist(name='ortrack', parameter_name='deit_tiny_patch16_224', dataset_name=dataset_name,
                            run_ids=None, display_name='ORTracK'))


dataset = get_dataset(dataset_name)

print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec', 'prec'))

