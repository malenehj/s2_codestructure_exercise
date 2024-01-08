import torch

if __name__ == '__main__':
    # Get the data and process it
    """Return train and test dataloaders for MNIST."""

    datapath = '/Users/fzv545/Documents/SODAS/PhD kurser/Machine Learning Operations (DTU)/s2_codestructure_exercise/data/'

    train_data, train_labels = [ ], [ ]
    for i in range(5):
        train_data.append(torch.load(datapath+f"raw/train_images_{i}.pt"))
        train_labels.append(torch.load(datapath+f"raw/train_target_{i}.pt"))

    train_data = torch.cat(train_data, dim=0)
    train_labels = torch.cat(train_labels, dim=0)

    test_data = torch.load(datapath+"raw/test_images.pt")
    test_labels = torch.load(datapath+"raw/test_target.pt")

    print(train_data.shape)
    print(train_labels.shape)
    print(test_data.shape)
    print(test_labels.shape)

    train_data = train_data.unsqueeze(1)
    test_data = test_data.unsqueeze(1)

    # Normalize 
    mean, std, var = torch.mean(train_data), torch.std(train_data), torch.var(train_data)
    train_data  = (train_data-mean)/std 

    mean, std, var = torch.mean(test_data), torch.std(test_data), torch.var(test_data)
    test_data  = (test_data-mean)/std

    train_set = torch.utils.data.TensorDataset(train_data, train_labels)
    test_set = torch.utils.data.TensorDataset(test_data, test_labels)

    torch.save(train_set,datapath+'processed/processed_images_train.pt')
    torch.save(test_set,datapath+'processed/processed_images_test.pt')
