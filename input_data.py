"""Functions for downloading and reading MNIST data."""
import gzip
import os
import urllib
import numpy
from scipy.misc import imread, imresize, toimage

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'


def maybe_download(filename, work_directory):
    """Download the data from Yann's website, unless it's already here."""
    if not os.path.exists(work_directory):
        os.mkdir(work_directory)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        filepath, _ = urllib.urlretrieve(SOURCE_URL + filename, filepath)
        statinfo = os.stat(filepath)
        print 'Succesfully downloaded', filename, statinfo.st_size, 'bytes.'
    return filepath


def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(filename):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    print 'Extracting', filename
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                'Invalid magic number %d in MNIST image file: %s' %
                (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(filename, one_hot=False):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    print 'Extracting', filename
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                'Invalid magic number %d in MNIST label file: %s' %
                (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8)
        if one_hot:
            return dense_to_one_hot(labels)
        return labels


class DataSet(object):
    def __init__(self, images, labels, fake_data=False):
        if fake_data:
            self._num_examples = 10000
        else:
            assert images.shape[0] == labels.shape[0], (
                "images.shape: %s labels.shape: %s" % (images.shape,
                                                       labels.shape))
            self._num_examples = images.shape[0]
            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns] (assuming depth == 1)
            assert images.shape[3] == 1
            images = images.reshape(images.shape[0],
                                    images.shape[1] * images.shape[2])
            # Convert from [0, 255] -> [0.0, 1.0].
            images = images.astype(numpy.float32)
            images = numpy.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1.0 for _ in xrange(784)]
            fake_label = 0
            return [fake_image for _ in xrange(batch_size)], [
                fake_label for _ in xrange(batch_size)]
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


def read_data_sets(train_dir, fake_data=False, one_hot=False):
    class DataSets(object):
        pass
    data_sets = DataSets()
    if fake_data:
        data_sets.train = DataSet([], [], fake_data=True)
        data_sets.validation = DataSet([], [], fake_data=True)
        data_sets.test = DataSet([], [], fake_data=True)
        return data_sets
    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
    TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
    TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
    VALIDATION_SIZE = 5000
    local_file = maybe_download(TRAIN_IMAGES, train_dir)
    train_images = extract_images(local_file)
    local_file = maybe_download(TRAIN_LABELS, train_dir)
    train_labels = extract_labels(local_file, one_hot=one_hot)
    local_file = maybe_download(TEST_IMAGES, train_dir)
    test_images = extract_images(local_file)
    local_file = maybe_download(TEST_LABELS, train_dir)
    test_labels = extract_labels(local_file, one_hot=one_hot)
    validation_images = train_images[:VALIDATION_SIZE]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_images = train_images[VALIDATION_SIZE:]
    train_labels = train_labels[VALIDATION_SIZE:]
    data_sets.train = DataSet(train_images, train_labels)
    data_sets.validation = DataSet(validation_images, validation_labels)
    data_sets.test = DataSet(test_images, test_labels)
    return data_sets

def get_data(num_classes=250 , res=128, flip=True, color_invert=False, center=False):
    """
    Generates the datasets with 128 (or 64) training examples, 8 validation examples,
    and 8 testing examples per class.

    Args:
        num_classes: the number of classes to load (for smaller datasets)
        res: the resolution of the output arrays (N x res x res)
        flip: whether or not to generate additional training examples by horizontally
              flipping the provided images
        color_invert: whether or not to invert B&W values
    """
    root_dir = "data/png{}/".format("" if res is None else res)

    num_train = 96 if flip else 48
    num_val = 16
    num_test = 16

    labels = []

    X_train = numpy.zeros((num_classes * num_train, res, res, 1), dtype=numpy.float32)
    y_train = numpy.repeat(numpy.arange(num_classes), num_train)

    X_val = numpy.zeros((num_classes * num_val, res, res, 1), dtype=numpy.float32)
    y_val = numpy.repeat(numpy.arange(num_classes), num_val)

    X_test = numpy.zeros((num_classes * num_test, res, res, 1), dtype=numpy.float32)
    y_test = numpy.repeat(numpy.arange(num_classes), num_test)

    classes = 0
    train_index = 0
    val_index = 0
    test_index = 0

    for node in sorted(os.listdir(root_dir)):
        if os.path.isfile(root_dir + node):
            continue

        labels.append(node)
        label_path = root_dir + node + "/"

        num_images = 0
        for im_file in sorted(os.listdir(label_path)):
            im_data = load_image(label_path + im_file).reshape(res, res, 1)

            if color_invert:
                im_data = -1 * im_data + 255

            if num_images < num_train:
                X_train[train_index] = im_data
                train_index += 1

                if flip:
                    X_train[train_index] = numpy.flip(im_data, axis=1)
                    train_index += 1
                    num_images += 1

            elif num_images < num_train + num_val:
                X_val[val_index] = im_data
                val_index += 1
            else:
                X_test[test_index] = im_data
                test_index += 1

            num_images += 1

        classes += 1
        if classes == num_classes:
            break

    if center:
        X_train -= numpy.mean(X_train, axis=0)
        X_val -= numpy.mean(X_val, axis=0)
        X_test -= numpy.mean(X_test, axis=0)
    return X_train, y_train, X_val, y_val, X_test, y_test, labels

def load_image(path):
    im_data = imread(path, mode='L')
    return im_data

def read_data_set_custom():
    class DataSets(object):
        pass
    data_sets = DataSets()
    X_train, y_train, X_val, y_val, X_test, y_test, labels = get_data()
    data_sets.train = DataSet(X_train, y_train)
    data_sets.validation = DataSet(X_val, y_val)
    data_sets.test = DataSet(X_test, y_test)
    return data_sets
