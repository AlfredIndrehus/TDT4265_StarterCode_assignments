import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer


def main():
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .1
    batch_size = 32
    neurons_per_layer_4a = [32, 10] #for task 4a
    neurons_per_layer_4b = [128, 10] #for task 4b
    momentum_gamma = .9  # Task 3 hyperparameter
    
    shuffle_data = True
    use_improved_sigmoid = False
    use_improved_weight_init = False
    use_momentum = False
    use_relu = False
    
    # Load dataset
    X_train1, Y_train1, X_val1, Y_val1 = utils.load_full_mnist()
    X_train1 = pre_process_images(X_train1)
    X_val1 = pre_process_images(X_val1)
    Y_train1 = one_hot_encode(Y_train1, 10)
    Y_val1 = one_hot_encode(Y_val1, 10)


    #Training 2 a model with task3 improvements, but different archtecture
    print("First model ")
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
        neurons_per_layer_4a,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    improvedtrainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        improvedmodel, learning_rate, batch_size, shuffle_data,
        X_train2, Y_train2, X_val2, Y_val2,
    )
    train_history1, val_history1 = improvedtrainer.train(num_epochs)


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
        neurons_per_layer_4b,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    improvedtrainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        improvedmodel, learning_rate, batch_size, shuffle_data,
        X_train2, Y_train2, X_val2, Y_val2,
    )
    train_history2, val_history2 = improvedtrainer.train(num_epochs)

   
    # Plotting the loss and accuracy (comment the above out)
    plt.subplot(1, 2, 1)  
    plt.ylim([0, 1])
    
    utils.plot_loss(train_history1["loss"], "Model without 32 hidden units", npoints_to_average=10)
    utils.plot_loss(train_history2["loss"], "Model with 128 hidden units", npoints_to_average=10)
   
    plt.ylabel("Training Loss")
    
    plt.subplot(1, 2, 2)  
    plt.ylim([0,1])
    
    utils.plot_loss(val_history1["accuracy"], "Model without 32 hidden units")
    utils.plot_loss(val_history2["accuracy"], "Model without 128 hidden unit")
    plt.ylabel("Validation Accuracy")

    plt.legend()
    plt.savefig("Task4a_and4b.png")
    plt.show()

    


if __name__ == "__main__":
    main()