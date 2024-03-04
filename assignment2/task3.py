import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, sigmoid, sigmoid_derivative, improved_sigmoid, improved_sigmoid_derivative, SoftmaxModel
from task2 import SoftmaxTrainer



def main():
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = 0.1
    batch_size = 32
    neurons_per_layer = [64, 10]
    neurons_per_layer_4a = [32, 10] #for task 4a
    neurons_per_layer_4b = [128, 10] #for task 4b
    momentum_gamma = .9  # Task 3 hyperparameter
    
    shuffle_data = True
    use_relu = False

    
    # Example created for comparing with and without shuffling.
    # For comparison, show all loss/accuracy curves in the same plot
    
    #1 - Training a model with weight initialization enabled 
    

    print("First model ")
    use_improved_sigmoid = False
    use_improved_weight_init = True
    use_momentum = False
    learning_rate = .1
    
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
    
    
    #2 - Training a model with improved sigmoid
    print("Second model ")
    
    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = False
    learning_rate = .1
    
    
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
    

    
    #3 - Training a model with all improvements enabled
    print("Third model")
    learning_rate = 0.02
    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True
    momentum_gamma1 = .9 
    
    
    # Load dataset
    X_train3, Y_train3, X_val3, Y_val3 = utils.load_full_mnist()
    X_train3 = pre_process_images(X_train3)
    X_val3 = pre_process_images(X_val3)
    Y_train3 = one_hot_encode(Y_train3, 10)
    Y_val3 = one_hot_encode(Y_val3, 10)
    
    finalmodel = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    finaltrainer= SoftmaxTrainer(
        momentum_gamma1, use_momentum,
        finalmodel, learning_rate, batch_size, shuffle_data,
        X_train3, Y_train3, X_val3, Y_val3,
    )
    train_history3, val_history3 = finaltrainer.train(num_epochs)

    #3 - Training a model without any improvements
    print ("Fourth model")
    learning_rate = 0.1
    use_improved_sigmoid = False
    use_improved_weight_init = False
    use_momentum = False

    # Load dataset
    X_train4, Y_train4, X_val4, Y_val4 = utils.load_full_mnist()
    X_train4 = pre_process_images(X_train4)
    X_val4 = pre_process_images(X_val4)
    Y_train4 = one_hot_encode(Y_train4, 10)
    Y_val4 = one_hot_encode(Y_val4, 10)

    finalmodel = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    finaltrainer= SoftmaxTrainer(
        momentum_gamma1, use_momentum,
        finalmodel, learning_rate, batch_size, shuffle_data,
        X_train4, Y_train4, X_val4, Y_val4,
    )
    train_history4, val_history4 = finaltrainer.train(num_epochs)


    

    # task 3
    # Plotting the loss and accuracy
    plt.figure(figsize=(20, 12))
    plt.subplot(1, 2, 1)  
    plt.ylim([0, 0.9])
    
    utils.plot_loss(train_history1["loss"], "Model with weight improvements", npoints_to_average=10)
    utils.plot_loss(train_history2["loss"], "Model with weight improvements and improved sigmoid", npoints_to_average=10)
    utils.plot_loss(train_history3["loss"], "Model with all task3 improvements", npoints_to_average=10)
    utils.plot_loss(train_history4["loss"], "Model without any improvements", npoints_to_average=10)
    plt.ylabel("Training Loss")
    
    plt.subplot(1, 2, 2)  
    plt.ylim([0.4,1])
    
    utils.plot_loss(val_history1["accuracy"], "Model with weight improvements")
    utils.plot_loss(val_history2["accuracy"], "Model with weight improvements and improved sigmoid")
    utils.plot_loss(val_history3["accuracy"], "Model with all task3 improvements")
    utils.plot_loss(val_history4["accuracy"], "Model without any improvements")
    plt.ylabel("Validation Accuracy")


    plt.legend()
    plt.savefig("task3c_train_loss.png")
    plt.show()


if __name__ == "__main__":
    main()
