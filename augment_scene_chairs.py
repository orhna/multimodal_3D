import numpy as np
from dataset_augmenter import DatasetAugmenter

path_2d_data = '/mnt/3dcv/projects/openscene_test/data/scannet/scannet_2d'
path_3d_data = '/mnt/3dcv/projects/openscene_test/data/scannet/scannet_3d'
path_output = '/home/aleks/3dcv/openseg_aug_new'

augmenter = DatasetAugmenter(path_2d_data, path_3d_data)

# STOOL
augmenter.load_model('9b975361717b4da79b18480bcc4e6dc8', 20,
                     rot_axis_angle=np.array([np.pi/2, 0, 0]),
                     height=0.6)

# ARMCHAIR
augmenter.load_model('cc480ea1a5974975af9fb23df1b181e5', 21,
                     rot_axis_angle=np.array([np.pi/2, 0, 0]),
                     height=1)
augmenter.rotate_model(21, np.array([0, 0, np.pi]))

# ROCKING CHAIR
augmenter.load_model('bcd3f4e05ec44902af3b5668b2b831ca', 22,
                     rot_axis_angle=np.array([np.pi/2, 0, 0]),
                     height=0.9)

# BALL CHAIR
augmenter.load_model('4363b93c1ea74caa8768bbcea426bb81', 23,
                     rot_axis_angle=np.array([np.pi/2, 0, 0]),
                     height=1.1)
augmenter.rotate_model(23, np.array([0, 0, -np.pi/2]))

augmenter.load_scene('scene0000_00')
augmenter.place_model_in_scene(np.array([4.6, 3.6]), 20)
augmenter.place_model_in_scene(np.array([3.15, 3.15]), 21)
augmenter.place_model_in_scene(np.array([1.6, 3.5]), 22)
augmenter.place_model_in_scene(np.array([6., 4.5]), 23)

augmenter.export(path_output)
