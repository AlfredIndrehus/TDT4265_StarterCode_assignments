
import pathlib
import matplotlib.pyplot as plt
import utils
import torch
from torch import nn
from dataloaders import load_cifar10
from trainer import Trainer, compute_loss_and_accuracy
from torchsummary import summary


##Task 3a
#Model from task2 copied as a starting point


class Task3Model_Old(nn.Module):
    def __init__(self, image_channels, num_classes):
        """
        Is called when model is initialized.
        Args:
            image_channels. Number of color channels in image (3)
            num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        # TODO: Implement this function (Task  2a)
        num_filters = 64  # Set number of filters in first conv layer
        self.num_classes = num_classes
        # Define the convolutional layers
        """
            After the first layer the output of feature_extractor would be [batch_size, num_filters, 32, 32]
            maxpool with stride=2 will half the size of the output (from 32L and 32 W to 16L and 16W)
            After the MaxPool2d layer the output of feature_extractor would be [batch_size, num_filters, 16, 16]
            that means after both the first convv and Pool layer  we would have:
            self.num_output_features = 32 * 16 * 16

            Formula for output size of conv layer:
            Width :
            W2 = [(W1 -FW + 2PW )/SW ] + 1
        """

        self.feature_extractor = nn.Sequential(
            #layer1
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=num_filters,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ELU(),
            #layer2
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            #layer3
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_filters*2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            
            nn.ELU(),
            #layer4
            nn.Conv2d(
                in_channels=num_filters*2,
                out_channels=num_filters*2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #layer5
            nn.Conv2d(
                in_channels=num_filters*2,
                out_channels=num_filters*4,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            
            nn.ELU(),
            #layer6
            nn.Conv2d(
                in_channels=num_filters*4,
                out_channels=num_filters*4,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #After this layer the output of feature_extractor would be [batch_size, num_filters * 4, 4, 4]

            #layer7
            nn.Conv2d(
                in_channels=num_filters*4,
                out_channels=num_filters*8,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            
            nn.ELU(),
            nn.Dropout(0.4), #p = 0.4 best score for now, with 64 filters

            #layer8
            nn.Conv2d(
                in_channels=num_filters*8,
                out_channels=num_filters*8,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        #The output of feature_extractor will be [batch_size, num_filters * 4, 4, 4]
        self.num_output_features = num_filters * 8 * 2 * 2  
        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class. 
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss

        self.classifier = nn.Sequential(
            nn.Flatten(),
            #layer9
            nn.Linear(self.num_output_features, num_filters * 2),
            nn.ELU(),
            #layer10
            nn.Linear(num_filters * 2, num_classes),

        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        # TODO: Implement this function (Task  2a)
        features = self.feature_extractor(x)
        #make sure to flatten/reshape/view inbetween Convolution(feature_extract) and fullyconnected (classification) 
        out = self.classifier(features)
        
        batch_size = x.shape[0]
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (
            batch_size,
            self.num_classes,
        ), f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out


def create_plots(trainer: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(
        trainer.train_history["loss"], label="Training loss", npoints_to_average=10
    )
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer.validation_history["accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()

