from base import BaseDataLoader

from .kitti_odometry_dataset import *

class KittiOdometryDataloader(BaseDataLoader):

    def __init__(self, batch_size=1, shuffle=True, validation_split=0.0, num_workers=4, **kwargs):
        self.dataset = KittiOdometryDataset(**kwargs)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)