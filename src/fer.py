# -*- coding: utf-8 -*-
"""FER2013 dataset.

Used as torch.utils.data.Dataset.
"""


import os
import pickle

import numpy as np
import pandas as pd
import PIL.Image
import torch


__all__ = ['FER2013', 'FER2013ReLU']
__author__ = 'Hao Zhang'
__copyright__ = '2018 LAMDA'
__date__ = '2018-05-09'
__email__ = 'zhangh0214@gmail.com'
__license__ = 'CC BY-SA 3.0'
__status__ = 'Development'
__updated__ = '2018-05-10'
__version__ = '1.0'


class FER2013(torch.utils.data.Dataset):
    """FER2013 Dataset.

    Args:
        _root, str: Root directory of dataset.
        _phase ['train'], str: train/val/test.
        _transform [None], function: A transform for a PIL.Image
        _target_transform [None], function: A transform for a label.

        _train_data, np.ndarray of shape N*3*48*48.
        _train_labels, np.ndarray of shape N.
        _val_data, np.ndarray of shape N*3*48*48.
        _val_labels, np.ndarray of shape N.
        _test_data, np.ndarray of shape N*3*48*48.
        _test_labels, np.ndarray of shape N.
    """
    def __init__(self, root, phase='train', transform=None,
                 target_transform=None):
        self._root = os.path.expanduser(root)
        self._phase = phase
        self._transform = transform
        self._target_transform = target_transform

        if (os.path.isfile(os.path.join(root, 'processed', 'train.pkl'))
            and os.path.isfile(os.path.join(root, 'processed', 'val.pkl'))
            and os.path.isfile(os.path.join(root, 'processed', 'test.pkl'))):
            print('Dataset already processed.')
        else:
            self.process('train', 28709)
            self.process('val', 3589)
            self.process('test', 3589)

        if self._phase == 'train':
            self._train_data, self._train_labels = pickle.load(
                open(os.path.join(self._root, 'processed', 'train.pkl'), 'rb'))
        elif self._phase == 'val':
            self._val_data, self._val_labels = pickle.load(
                open(os.path.join(self._root, 'processed', 'val.pkl'), 'rb'))
        elif self._phase == 'test':
            self._test_data, self._test_labels = pickle.load(
                open(os.path.join(self._root, 'processed', 'test.pkl'), 'rb'))
        else:
            raise ValueError('phase should be train/val/test.')

    def __getitem__(self, index):
        """Fetch a particular example (X, y).

        Args:
            index, int.

        Returns:
            image, torch.Tensor.
            label, int.
        """
        if self._phase == 'train':
            image, label = self._train_data[index], self._train_labels[index]
        elif self._phase == 'val':
            image, label = self._val_data[index], self._val_labels[index]
        elif self._phase == 'test':
            image, label = self._test_data[index], self._test_labels[index]
        else:
            raise ValueError('phase should be train/val/test.')

        image = PIL.Image.fromarray(image.astype('uint8'))
        if self._transform is not None:
            image = self._transform(image)
        if self._target_transform is not None:
            label = self._target_transform(label)

        return image, label

    def __len__(self):
        """Dataset length.

        Returns:
            length, int.
        """
        if self._phase == 'train':
            return len(self._train_data)
        elif self._phase == 'val':
            return len(self._val_data)
        elif self._phase == 'test':
            return len(self._test_data)
        else:
            raise ValueError('phase should be train/val/test.')

    def process(self, phase, size):
        """Fetch train/val/test data from raw csv file and save them onto
        disk.

        Args:
            phase, str: 'train'/'val'/'test'.
            size, int. Size of the dataset.
        """
        if phase not in ['train', 'val', 'test']:
            raise ValueError('phase should be train/val/test')
        # Load all data.
        print('Processing dataset.')
        data_frame = pd.read_csv(os.path.join(
            self._root, 'raw', '%s_all.csv' % phase))

        # Fetch all labels.
        labels = data_frame['emotion'].values  # np.ndarray
        assert labels.shape == (size,)

        # Fetch all images.
        data_frame['pixels'].to_csv(
            os.path.join(self._root, 'build', '%s_image.csv' % phase),
            header=None, index=False)
        data_frame = pd.read_csv(
            os.path.join(self._root, 'build', '%s_image.csv' % phase),
            index_col=None, delim_whitespace=True, header=None)

        images = data_frame.values.astype('float64')
        assert images.shape == (size, 48 * 48)
        images = images.reshape(size, 48, 48, 1)

        images = np.concatenate((images, images, images), axis=3)
        assert images.shape == (size, 48, 48, 3)

        pickle.dump(
            (images, labels),
            open(os.path.join(self._root, 'processed', '%s.pkl' % phase), 'wb'))


class FER2013ReLU(torch.utils.data.Dataset):
    """FER2013 relu5-3 Dataset.

    Args:
        _root, str: Root directory of dataset.
        _phase ['train'], str: train/val/test.

        _train_data, list<torch.Tensor>.
        _train_labels, list<int>.
        _val_data, list<torch.Tensor>.
        _val_labels, list<int>.
        _test_data, list<torch.Tensor>.
        _test_labels, list<int>.
    """
    def __init__(self, root, phase='train'):
        self._root = os.path.expanduser(root)
        self._phase = phase

        if (os.path.isfile(os.path.join(root, 'relu5-3', 'train.pkl'))
            and os.path.isfile(os.path.join(root, 'relu5-3', 'val.pkl'))
            and os.path.isfile(os.path.join(root, 'relu5-3', 'test.pkl'))):
            print('Dataset already processed.')
        else:
            raise RuntimeError('FER-2013 relu5-3 Dataset not found.'
                'You need to prepare it in advance.')

        # Now load the picked data.
        if self._phase == 'train':
            self._train_data, self._train_labels = pickle.load(
                open(os.path.join(self._root, 'relu5-3', 'train.pkl'), 'rb'))
            assert (len(self._train_data) == 28709
                    and len(self._train_labels) == 28709)
        elif self._phase == 'val':
            self._val_data, self._val_labels = pickle.load(
                open(os.path.join(self._root, 'relu5-3', 'val.pkl'), 'rb'))
            assert (len(self._val_data) == 3589
                    and len(self._val_labels) == 3589)
        elif self._phase == 'test':
            self._test_data, self._test_labels = pickle.load(
                open(os.path.join(self._root, 'relu5-3', 'test.pkl'), 'rb'))
            assert (len(self._test_data) == 3589
                    and len(self._test_labels) == 3589)
        else:
            raise ValueError('phase should be train/val/test.')

    def __getitem__(self, index):
        """Fetch a particular example (X, y).

        Args:
            index, int.

        Returns:
            image, torch.Tensor.
            label, int.
        """
        if self._phase == 'train':
            image, label = self._train_data[index], self._train_labels[index]
        elif self._phase == 'val':
            image, label = self._val_data[index], self._val_labels[index]
        elif self._phase == 'test':
            image, label = self._test_data[index], self._test_labels[index]
        else:
            raise ValueError('phase should be train/val/test.')

        return image, label

    def __len__(self):
        """Dataset length.

        Returns:
            length, int.
        """
        if self._phase == 'train':
            return len(self._train_data)
        elif self._phase == 'val':
            return len(self._val_data)
        elif self._phase == 'test':
            return len(self._test_data)
        else:
            raise ValueError('phase should be train/val/test.')
