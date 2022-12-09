class NeuralNetwork :
    def __init__(self) :
        self.layers = []
        self.loss = None
        self.lossPrime = None
    
    # method to add a layer to the network
    def add(self, layer) :
        self.layers.append(layer)
    
    # method to set loss function
    def setLossFunction(self, loss) :
        self.loss = loss
    
    # method to train the network 
    def fit(self, X_train, y_train, epochs, learning_rate) :
        samples = len(X_train)

        # training loop
        for i in range(epochs) :
            print("\nStart of epoch %d" % (i+1))

            err = 0
            for j in range(samples) :
                # carry out forward propagation
                output = X_train[j]
                for layer in self.layers :
                    output = layer.forwardPropagation(output)
                
                # calculate the loss
                err += self.loss(y_train[j], output)

                # carry out back-propagation
                error = self.loss(y_train[j], output, derivative = True)
                for layer in reversed(self.layers):
                    error = layer.backPropagation(error, learning_rate)
            
            # calculate average error on all samples
            err /= samples
            
            print("Error = %f" % err)
    
    # method to predict the output
    def predict(self, input_data) :
        samples = len(input_data)
        result = []

        # interate over all samples 
        for i in range(samples) :
            # carry out forward propagation
            output = input_data[i]
            for layer in self.layers :
                output = layer.forwardPropagation(output)
            result.append(output)
        
        return result