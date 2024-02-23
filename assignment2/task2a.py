import numpy as np
import utils
import typing

np.random.seed(1)


def pre_process_images(X: np.ndarray):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785] normalized as described in task2a
    """
    assert X.shape[1] == 784, f"X.shape[1]: {X.shape[1]}, should be 784"
    # TODO implement this function (Task 2a)
 
    mean = np.mean(X)
    std = np.std(X)
    X_norm = (X-mean)/(std)

    #bias trick
    bias_column = np.ones((X_norm.shape[0], 1))
    X_norm = np.concatenate((X_norm, bias_column), axis=1)
    
    assert X_norm.shape[1] == 785, f"X.shape[1]: {X.shape[1]}, should be 785 (after bias trick)"
    return X_norm



def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray):
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    Returns:
        Cross entropy error (float)
    """
    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"
    cross_entropy = -np.sum(targets*np.log(outputs), axis=1)
    return np.mean(cross_entropy)

 
#Task 3b) defining improved sigmoid function and derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def improved_sigmoid(x):
    return 1.7159 * np.tanh(2/3 * x)
def sigmoid_derivative(x):
    return x * (1 - x)
def improved_sigmoid_derivative(x):
    return 1.7159 * 2/3 * (1 - np.tanh(2/3 * x)**2)

class SoftmaxModel:
    
   

    def __init__(
        self,
        # Number of neurons per layer
        neurons_per_layer: typing.List[int],
        use_improved_sigmoid: bool,  # Task 3b hyperparameter
        use_improved_weight_init: bool,  # Task 3a hyperparameter
        use_relu: bool,  # Task 3c hyperparameter
    ):
        np.random.seed(
            1
        )  # Always reset random seed before weight init to get comparable results.
        # Define number of input nodes
        self.I = 785
        self.use_improved_sigmoid = use_improved_sigmoid
        self.use_relu = use_relu
        self.use_improved_weight_init = use_improved_weight_init
        
        

        # Define number of output nodes
        # neurons_per_layer = [64, 10] indicates that we will have two layers:
        # A hidden layer with 64 neurons and a output layer with 10 neurons.
        self.neurons_per_layer = neurons_per_layer


        # Initialize the weights
        self.ws = []
        prev = self.I
        for size in self.neurons_per_layer:
            w_shape = (prev, size)
            print("Initializing weight to shape:", w_shape)
            #Task 3a) improved weight init. Checks if use_improved_weight_init is true.
            # If not, use normal weight init with uniform distribution. 
            if self.use_improved_weight_init == True: #For some reason this is not working unless I use == True ?? 
                w = np.random.normal(0, 1/np.sqrt(prev), size=w_shape)
                #task 2c) initialize weight to random
            else:
                w = np.random.uniform(-1, 1, size=w_shape)
            self.ws.append(w)
            prev = size
        self.grads = [None for i in range(len(self.ws))]
        
        #task 3b) improved sigmoid init. Checks if use_improved_sigmoid is true.
        # If not, use normal sigmoid function.
        
        if self.use_improved_sigmoid:
            self.sigmoid = improved_sigmoid
            self.sigmoid_derivative = improved_sigmoid_derivative
        else:
            self.sigmoid = sigmoid
            self.sigmoid_derivative = sigmoid_derivative
                    
        
        
        #new ones;
        #self.hidden_layer_output = []
        #self.output_layer_outputs = []
        #self.neurons_hidden_layer = self.neurons_per_layer[0]
        #self.neurons_output_layer = self.neurons_per_layer[1]

        # task4
        self.z = []
        self.a = []


    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """

        # TODO implement this function (Task 2b)
        # HINT: For performing the backward pass, you can save intermediate activations in variables in the forward pass.
        # such as self.hidden_layer_output = ...
        batch_size= X.shape[0]

        #input to hidden layer.
        self.z.append(np.dot(X, self.ws[0]))
        #outp0ut from hidden layer (using sigmoid func), one hidden layer
        self.a.append(self.sigmoid(self.z[0]))
        number_hiddenlayers = len(self.neurons_per_layer) -1
        for i in range(1, number_hiddenlayers):
            self.z.append(np.dot(self.a[i-1], self.ws[i]))
            self.a.append(self.sigmoid(z[i]))


        #input to outputlayer
        self.z.append(np.dot(self.a[-1], self.ws[1]))
        #outout from outputlayer = predictions
        self.a.append(np.exp(self.z[-1]) / np.sum(np.exp(self.z[-1]), axis=1, keepdims=True))
        #save it to self
        
        """
        #input to hidden layer.
        z2= np.dot(X, self.ws[0])
        #output from hidden layer (using sigmoid func), one hidden layer
        a2= self.sigmoid(z2)
        
        #output from hidden layer (using sigmoid func)
        #a2= 1 / (1 + np.exp(z2))
        
        #save it to self for use in backward function
        self.hidden_layer_output=a2

        #input to outputlayer
        z3 = np.dot(a2, self.ws[1])
        #outout from outputlayer = predictions
        a3 = np.exp(z3) / np.sum(np.exp(z3), axis=1, keepdims=True)
        #save it to self
        self.output_layer_outputs = a3
        """
        #self.output_layer_outputs = self.a[-1]
        #self.hidden_layer_output = self.a[-2]
        return self.a[-1]

    def backward(self, X: np.ndarray, outputs: np.ndarray, targets: np.ndarray) -> None:
        """
        Computes the gradient and saves it to the variable self.grad

        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        """
        # TODO implement this function (Task 2b)
        assert (
            targets.shape == outputs.shape
        ), f"Output shape: {outputs.shape}, targets: {targets.shape}"
        # A list of gradients.
        # For example, self.grads[0] will be the gradient for the first hidden layer
        
        batch_size = X.shape[0]

        #output layer
        error_output = outputs - targets
        self.grads[1] = np.dot(self.a[-2].T, error_output) / batch_size
        

        #hidden layer (using chain rule)
        error_hidden = np.dot(error_output, self.ws[1].T)
        d_sigmoid_hidden = self.sigmoid_derivative(self.a[-2])
        #d_sigmoid_hidden = self.hidden_layer_output * (1- self.hidden_layer_output)
        grad_hidden = error_hidden * d_sigmoid_hidden
        self.grads[0] = np.dot(X.T, grad_hidden) / (batch_size)


        for grad, w in zip(self.grads, self.ws):
            assert (
                grad.shape == w.shape
            ), f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."



    def zero_grad(self) -> None:
        self.grads = [None for i in range(len(self.ws))]


def one_hot_encode(Y: np.ndarray, num_classes: int):
    """
    Args:
        Y: shape [Num examples, 1]
        num_classes: Number of classes to use for one-hot encoding
    Returns:
        Y: shape [Num examples, num classes]
    """
    # TODO: Implement this function (copy from last assignment)
  
    Y_onehotencoded= np.zeros((Y.shape[0],num_classes))
    for i in range(Y.shape[0]):
        Y_onehotencoded[i, int(Y[i])]=1
    
    return Y_onehotencoded


def gradient_approximation_test(model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
    Numerical approximation for gradients. Should not be edited.
    Details about this test is given in the appendix in the assignment.
    """

    assert isinstance(X, np.ndarray) and isinstance(
        Y, np.ndarray
    ), f"X and Y should be of type np.ndarray!, got {type(X), type(Y)}"

    epsilon = 1e-3
    for layer_idx, w in enumerate(model.ws):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                orig = model.ws[layer_idx][i, j].copy()
                model.ws[layer_idx][i, j] = orig + epsilon
                logits = model.forward(X)
                cost1 = cross_entropy_loss(Y, logits)
                model.ws[layer_idx][i, j] = orig - epsilon
                logits = model.forward(X)
                cost2 = cross_entropy_loss(Y, logits)
                gradient_approximation = (cost1 - cost2) / (2 * epsilon)
                model.ws[layer_idx][i, j] = orig
                # Actual gradient
                logits = model.forward(X)
                model.backward(X, logits, Y)
                difference = gradient_approximation - \
                    model.grads[layer_idx][i, j]
                assert abs(difference) <= epsilon**1, (
                    f"Calculated gradient is incorrect. "
                    f"Layer IDX = {layer_idx}, i={i}, j={j}.\n"
                    f"Approximation: {gradient_approximation}, actual gradient: {model.grads[layer_idx][i, j]}\n"
                    f"If this test fails there could be errors in your cross entropy loss function, "
                    f"forward function or backward function"
                )


def main():
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert (
        Y[0, 3] == 1 and Y.sum() == 1
    ), f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    Y_train = one_hot_encode(Y_train, 10)
    assert (
        X_train.shape[1] == 785
    ), f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    neurons_per_layer = [64, 10]
    use_improved_sigmoid = False
    use_improved_weight_init = True
    use_relu = True
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init, use_relu
    )

    
    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    gradient_approximation_test(model, X_train, Y_train)


if __name__ == "__main__":
    main()