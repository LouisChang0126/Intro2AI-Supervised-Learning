import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.utils import shuffle
from loguru import logger
from sklearn.metrics import accuracy_score

from decision_tree import ConvNet, DecisionTree, get_features_and_labels
from resnet import ResNet, train, validate
from utils import myDataset, load_train_dataset, load_test_dataset

"""
Notice:
    1) You can't add any additional package
    2) You don't have to change anything in main()
    3) You can ignore the suggested data type if you want
"""

def main():
    """
    load data
    """
    logger.info("Start loading data")
    images, labels = load_train_dataset()
    images, labels = shuffle(images, labels, random_state=777)
    train_len = int(0.8 * len(images))

    train_images, test_images = images[:train_len], images[train_len:]
    train_labels, test_labels = labels[:train_len], labels[train_len:]

    train_dataset = myDataset(train_images, train_labels)
    val_dataset = myDataset(test_images, test_labels)

    """
    Decision Tree
    """
    logger.info("Start training Decision Tree")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize Models
    conv_model = ConvNet().to(device)
    tree_model = DecisionTree(max_depth=7)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Extract features from training data
    train_features, train_labels = get_features_and_labels(conv_model, train_loader, device)

    # Train the decision tree classifier
    tree_model.fit(train_features, train_labels)

    # Extract features from validation data
    val_features, val_labels = get_features_and_labels(conv_model, val_loader, device)

    # Predict with the trained decision tree
    val_predictions = tree_model.predict(val_features)

    # Calculate accuracy
    val_accuracy = accuracy_score(val_labels, val_predictions)
    logger.info(f"Validation Accuracy: {val_accuracy:.4f}")

    """
    Resnet
    """
    logger.info("Start training ResNet")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = ResNet().to(device)
    criterion = nn.CrossEntropyLoss()

    # Optimizer configuration
    base_params = [param for name, param in model.named_parameters() if param.requires_grad]
    optimizer = optim.Adam(base_params, lr=1e-4)

    train_losses = []
    val_losses = []
    max_acc = 0

    EPOCHS = 10
    for epoch in range(EPOCHS): #epoch
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Acc:  {val_acc:.4f}")

        if val_acc > max_acc:
            print(f"Saving model, Best Accuracy: {val_acc:.4f}")
            torch.save(model.state_dict(), f'model.pth')
            max_acc = val_acc

    logger.info(f"Best Accuracy: {max_acc:.4f}")


if __name__ == '__main__':
    main()
