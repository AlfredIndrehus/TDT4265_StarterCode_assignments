import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer


def main():
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .1
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    
    shuffle_data = True
    use_improved_sigmoid = False
    use_improved_weight_init = False
    use_momentum = False
    use_relu = False

    # Load dataset
    """
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)
    """
    # Example created for comparing with and without shuffling.
    # For comparison, show all loss/accuracy curves in the same plot
    
    #1 - Training a model with weight initialization enabled 
    
    print("First model ")
    use_improved_sigmoid = False
    use_improved_weight_init = False
    use_momentum = False
    
    
    # Load dataset 
    X_train1, Y_train1, X_val1, Y_val1 = utils.load_full_mnist()
    X_train1 = pre_process_images(X_train1)
    X_val1 = pre_process_images(X_val1)
    Y_train1 = one_hot_encode(Y_train1, 10)
    Y_val1 = one_hot_encode(Y_val1, 10)

    
    weightmodel = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    weighttrainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        weightmodel, learning_rate, batch_size, shuffle_data,
        X_train1, Y_train1, X_val1, Y_val1,
    )
    train_history1, val_history1 = weighttrainer.train(num_epochs)
    
    
    #2 - Training a model with task3 improvements
    print("Second model ")
    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True
    learning_rate = 0.02
    
    # Load dataset
    X_train2, Y_train2, X_val2, Y_val2 = utils.load_full_mnist()
    X_train2 = pre_process_images(X_train2)
    X_val2 = pre_process_images(X_val2)
    Y_train2 = one_hot_encode(Y_train2, 10)
    Y_val2 = one_hot_encode(Y_val2, 10)
    
    improvedmodel = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    improvedtrainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        improvedmodel, learning_rate, batch_size, shuffle_data,
        X_train2, Y_train2, X_val2, Y_val2,
    )
    train_history2, val_history2 = improvedtrainer.train(num_epochs)
    
   
    
    
    
    # YOU CAN DELETE EVERYTHING BELOW!
    """
    shuffle_data = False

    # Train a new model with new parameters
    model_no_shuffle = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer_shuffle = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_no_shuffle, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_no_shuffle, val_history_no_shuffle = trainer_shuffle.train(
        num_epochs)
    """
    
    # Plotting the loss and accuracy
    
    plt.subplot(1, 2, 1)  
    plt.ylim([0, 1])
    
    utils.plot_loss(train_history1["loss"], "Model without task3 improvements", npoints_to_average=10)
    utils.plot_loss(train_history2["loss"], "Model with task3 improvements", npoints_to_average=10)
   
    plt.ylabel("Training Loss")
    
    plt.subplot(1, 2, 2)  
    plt.ylim([0,1])
    
    utils.plot_loss(val_history1["accuracy"], "Model without task3 improvements")
    utils.plot_loss(val_history2["accuracy"], "Model with task3 improvements")
    plt.ylabel("Validation Accuracy")


    plt.legend()
    plt.savefig("task3c_train_loss.png")
    plt.show()

        


if __name__ == "__main__":
    main()
