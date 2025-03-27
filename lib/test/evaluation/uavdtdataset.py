import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os


class UAVDTDataset(BaseDataset):
    """ UAVDT dataset.
    """
    def __init__(self):
        super().__init__()

        self.base_path = os.path.join(self.env_settings.uavdt_path)
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/{}/{}_gt.txt'.format(self.base_path, 'anno',sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        frames_path = '{}/{}/{}'.format(self.base_path, 'sequences',sequence_name)
        frame_list = [frame for frame in os.listdir(frames_path) if frame.endswith(".jpg")]
        frame_list.sort(key=lambda f: int(f[3:-4]))
        frames_list = [os.path.join(frames_path, frame) for frame in frame_list]
        return Sequence(sequence_name, frames_list, 'uavdt', ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        seq_path = os.path.join(self.base_path,'sequences')
        seqs = os.listdir(seq_path)
        return seqs
