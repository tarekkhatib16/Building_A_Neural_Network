# Building and Productionising a Neural Network from Scratch!

This projects explores building a traditional neural network from scratch with just Numpy. This is then followed by productionising the model by implementing teting, CI/CD, containerising using Docker, and surfacing using FastAPI.

## How It's Made

**Tech Used:** Python, FastAPI, Docker, Poetry, GitHub Actions

__Neural Network__ 

Building the neural network started with exploring the fundamental mathematics behind a neural network, such as what backpropagation was. I then started building the functions that would be needed such as the Sigmoid and Loss functions. Building the Layer class came next where I ensured that each layer had associated weights, biases, and an activation function, alongside a forward propagation and a backwards propagation method. This was all then brought together to predict the MNIST dataset which I retrieved from Kaggle (https://www.kaggle.com/datasets/oddrationale/mnist-in-csv).

__Testing__

Testing was carried out using unittest. I implemented unit and E2E tests. Unit tests ensured that all the individual parts of the project worked as expected, while the E2E tests ensured that each part interacts with the other as expected. 

__CI/CD Workflows__ 

Here I implemented two GitHub Actions workflows: 
    - CI: installed the dependecies and set up the environment then ran the unit tests and the E2E tests before creating a pull request from the dev branch into main which then needs to be approved on the GitHub UI. 
    - CD: once a change is merged with main, this workflow rebuilds the Docker container and redeploys the FastAPI app to incorporate any new changes that were implemented. 

__Docker__

The poetry dependencies were exported into a requirements.txt file. The Dockerfile has a simple set up that sets up the working directory before copying over and installing the dependencies, copies over the necessary files, exposes port 8000 for the web application, and starts the app using uvicorn ASGI. 

__FastAPI__

Here, on start up the model is loaded. Once predict is requested, the input JSON is reshaped before being passed into the model. The output is then decoded and outputted back as an integer. 

## Lessons Learnt & Possible Improvements

A Dockerfile is very difficult to implement using poetry and is something that needs a lot of work, the next iteration of the Dockerfile should have the environment set up directly from poetry instead of having to export poetry into a requirements.txt file. 

Ensuring the input data is in the correct format is difficult especially for data such as the MNIST dataset. This proved difficult when implementing testing and having to come up with sample data points, but also while setting up the app as it needed to take in a JSON. 

In terms of the model itself, understanding the fundamentals of how a simple neural network works has been essential in then building on that knowledge. Understanding how activation functions work means that I now have a deeping understanding of how to tailor each layer for a specific role within the network. Additionally, focusing on gradient descent as an optimiser opens up other optimisers that can be used in the future, such as ADAM or SGD. Finally, trialling out learning rates, batches, epochs, etc. would all allow the further optimisation of these values.

Using a well-known dataset such as MNIST removes much of the important work regarding data optimisation and feature engineering. In the future encorporating more of this would give me a deeper understanding of how real-life data can be fed into a neural network for predictions.

Finally, were this a fully productionised model, incorporating data and concept drift monitoring is important. 