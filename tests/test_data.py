from tests import _PATH_DATA
import torch
import pytest

import os.path


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_data_shape():

    train = torch.load(_PATH_DATA+'/processed/processed_images_train.pt')
    test = torch.load(_PATH_DATA+'/processed/processed_images_test.pt')

    N_train = 25000
    N_test = 5000

    assert len(train) == N_train, "The training data is not the right length"
    assert len(test) == N_test, "The test data is not the right length"
    assert train[0][0].shape == torch.Size([1,28,28]), "The first datapoint of the train data is the wrong shape" #just checking the first instance. Check all?
    assert test[0][0].shape == torch.Size([1,28,28]), "The first datapoint of the test data is the wrong shape"


def test_labels_present(): 
    train = torch.load(_PATH_DATA+'/processed/processed_images_train.pt')
    test = torch.load(_PATH_DATA+'/processed/processed_images_test.pt')

    # Check that all labels are present in the train subset 
    # Get all labels in int format
    all_labels = [data[1].item() for data in train]
    # Create a set of unique labels
    unique_labels = set(all_labels)

    #assert that all labels are represented
    # Check if all labels (0 to 9) are present
    assert len(unique_labels) == 10, "Not all labels are present in the dataset"
    for label in range(10):
        assert label in unique_labels, f"Label {label} is missing from the dataset"


    # Check that all labels are present in the test subset 
    # Get all labels in int format
    all_labels = [data[1].item() for data in test]
    # Create a set of unique labels
    unique_labels = set(all_labels)

    #assert that all labels are represented
    # Check if all labels (0 to 9) are present
    assert len(unique_labels) == 10, "Not all labels are present in the dataset"
    for label in range(10):
        assert label in unique_labels, f"Label {label} is missing from the dataset"


