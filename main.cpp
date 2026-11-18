// main.cpp
// Copyright 2026 Techspeck, LLC. All rights reserved.
// Neural Network Implementation
// This implementation is a simple feedforward neural network with two hidden layers.
// It uses ReLU activation for hidden layers and Sigmoid activation for the output layer.
// The network is trained using backpropagation and gradient descent.
// The network is designed to classify points as inside or outside a unit circle based on their coordinates.
// The network architecture is as follows:
// Input Layer: 2 neurons (x, y coordinates)
// Hidden Layer 1: 8 neurons, ReLU activation
// Hidden Layer 2: 4 neurons, ReLU activation
// Output Layer: 1 neuron, Sigmoid activation
// The training data consists of random points in the range [-2, 2] for both x and y coordinates.
// The target output is 1 if the point is inside the unit circle (distance from origin < 1), and 0 otherwise.
// You can adjust the number of epochs and learning rate to improve training performance.
// You can also modify the architecture by changing the number of neurons in the hidden layers.
// The code is written in C++ and uses the C++ standard library for random number generation, timing, and mathematical operations.
// Youtube Tutorial: https://www.https://www.youtube.com/watch?v=fYjH0HgqzfM

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <print>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace Activation {
inline double relu(double x) { return std::max(0.0, x); }
inline double reluDerivative(double x) { return (x > 0) ? 1.0 : 0.0; }
inline double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
inline double sigmoidDerivative(double x) {
    double s = sigmoid(x);
    return s * (1.0 - s);
}
}

class Matrix{
 private:
    std::vector<std::vector<double>> data;
    size_t rows, cols;

 public:
    Matrix(size_t r, size_t c):rows(r), cols(c) {
        data.resize(r, std::vector<double>(c, 0.0));
    }

    double &operator()(size_t i, size_t j) { return data[i][j]; }
    const double &operator()(size_t i, size_t j) const { return data[i][j]; }

    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; }
};

class NeuralNetwork {
 private:
    std::vector<int> layerSizes;
    Matrix weights1, weights2, weights3;
    std::vector<double> bias1, bias2, bias3;
    std::mt19937 gen;

    void initializeWeights() {
        std::normal_distribution<> dist(0.0, 1.0);
        double scale1 = sqrt(2.0 / layerSizes[0]);
        double scale2 = sqrt(2.0 / layerSizes[1]);
        double scale3 = sqrt(2.0 / layerSizes[2]);

        for (int i = 0; i < weights1.getRows(); i++) {
            for (int j = 0; j < weights1.getCols(); j++) {
                weights1(i, j) = dist(gen) * scale1;
            }
        }

        for (int i = 0; i < weights2.getRows(); i++) {
            for (int j = 0; j < weights2.getCols(); j++) {
                weights2(i, j) = dist(gen) * scale2;
            }
        }

        for (int i = 0; i < weights3.getRows(); i++) {
            for (int j = 0; j < weights3.getCols(); j++) {
                weights3(i, j) = dist(gen) * scale3;
            }
        }

        fill(bias1.begin(), bias1.end(), 0.0);
        fill(bias2.begin(), bias2.end(), 0.0);
        fill(bias3.begin(), bias3.end(), 0.0);
    }

 public:
    NeuralNetwork(int inputSize, int hidden1Size, int hidden2Szie, int outputSize)
    : layerSizes{inputSize,
        hidden1Size,
        hidden2Szie,
        outputSize},
        weights1(inputSize, hidden1Size),
        weights2(hidden1Size, hidden2Szie),
        weights3(hidden2Szie, outputSize),
        bias1(hidden1Size),
        bias2(hidden2Szie),
        bias3(outputSize),
        gen(std::random_device {}()) {
            if (inputSize <= 0 || hidden1Size <= 0 || hidden2Szie <= 0 || outputSize <= 0) {
                throw std::invalid_argument("Layer sizes must be positive !");
        }

        initializeWeights();
    }

    std::vector<double> forward(const std::vector<double> &input) {
        if (input.size() != layerSizes[0]) {
            throw std::runtime_error("Input size mismatch !");
        }

        std::vector<double> hidden1(layerSizes[1]);
        for (int j = 0; j < layerSizes[1]; j++) {
            double sum = bias1[j];
            for (int i = 0; i < layerSizes[0]; i++) {
                sum += input[i] * weights1(i, j);
            }

            hidden1[j] = Activation::relu(sum);
        }

        std::vector<double> hidden2(layerSizes[2]);
        for (int j = 0; j < layerSizes[2]; j++) {
            double sum = bias2[j];
            for (int i = 0; i < layerSizes[1]; i++) {
                sum += hidden1[i] * weights2(i, j);
            }

            hidden2[j] = Activation::relu(sum);
        }

        std::vector<double> output(layerSizes[3]);
        for (int j = 0; j < layerSizes[3]; j++) {
            double sum = bias3[j];
            for (int i = 0; i < layerSizes[2]; i++) {
                sum += hidden2[i] * weights3(i, j);
            }

            output[j] = Activation::sigmoid(sum);
        }

        return output;
    }

    void train(const std::vector<std::vector<double>> &inputs,
                const std::vector<std::vector<double>> &targets,
                double learningRate ,
                int epochs) {
        if (inputs.size() != targets.size()) {
            throw std::runtime_error("Input and Target sizes don't match !");
        }

        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalError = 0.0;

            for (size_t k = 0; k < inputs.size(); k++) {
                std::vector<double> hidden1(layerSizes[1]);
                std::vector<double> hidden1Pre(layerSizes[1]);
                for (int j = 0; j < layerSizes[1]; j++) {
                    double sum = bias1[j];
                    for (int i = 0; i < layerSizes[0]; i++) {
                        sum += inputs[k][i] * weights1(i, j);
                    }
                    hidden1Pre[j] = sum;
                    hidden1[j] = Activation::relu(sum);
                }

                std::vector<double> hidden2(layerSizes[2]);
                std::vector<double> hidden2Pre(layerSizes[2]);
                for (int j = 0; j < layerSizes[2]; j++) {
                    double sum = bias2[j];
                    for (int i = 0; i < layerSizes[1]; i++) {
                        sum += hidden1[i] * weights2(i, j);
                    }
                    hidden2Pre[j] = sum;
                    hidden2[j] = Activation::relu(sum);
                }

                std::vector<double> output(layerSizes[3]);
                std::vector<double> outputPre(layerSizes[3]);
                for (int j = 0; j < layerSizes[3]; j++) {
                    double sum = bias3[j];
                    for (int i = 0; i < layerSizes[2]; i++) {
                        sum += hidden2[i] * weights3(i, j);
                    }
                    outputPre[j] = sum;
                    output[j] = Activation::sigmoid(sum);
                }

                for (int j = 0; j < layerSizes[3]; j++) {
                    double error = targets[k][j] - output[j];
                    totalError += error * error;
                }

                std::vector<double> outputGradients(layerSizes[3]);
                for (int j = 0; j < layerSizes[3]; j++) {
                    outputGradients[j] = (output[j] - targets[k][j]) *
                                         Activation::sigmoidDerivative(outputPre[j]);
                }

                std::vector<double> hidden2Gradients(layerSizes[2]);
                for (int i = 0; i < layerSizes[2]; i++) {
                    double error = 0;
                    for (int j = 0; j < layerSizes[3]; j++) {
                        error += outputGradients[j] * weights3(i, j);
                    }

                    hidden2Gradients[i] = error * Activation::reluDerivative(hidden2Pre[i]);
                }

                std::vector<double> hidden1Gradients(layerSizes[1]);
                for (int i = 0; i < layerSizes[1]; i++) {
                    double error = 0;
                    for (int j = 0; j < layerSizes[2]; j++) {
                        error += hidden2Gradients[j] * weights2(i, j);
                    }

                    hidden1Gradients[i] = error * Activation::reluDerivative(hidden1Pre[i]);
                }

                for (int i = 0; i < layerSizes[2]; i++) {
                    for (int j = 0; j < layerSizes[3]; j++) {
                        weights3(i, j) -= learningRate * outputGradients[j] * hidden2[i];
                    }
                }

                for (int j = 0; j < layerSizes[3]; j++) {
                    bias3[j] -= learningRate * outputGradients[j];
                }

                for (int i = 0; i < layerSizes[1]; i++) {
                    for (int j = 0; j < layerSizes[2]; j++) {
                        weights2(i, j) -= learningRate * hidden2Gradients[j] * hidden1[i];
                    }
                }

                for (int j = 0; j < layerSizes[2]; j++) {
                    bias2[j] -= learningRate * hidden2Gradients[j];
                }

                for (int i = 0; i < layerSizes[0]; i++) {
                    for (int j = 0; j < layerSizes[1]; j++) {
                        weights1(i, j) -= learningRate * hidden1Gradients[j] * inputs[k][i];
                    }
                }

                for (int j = 0; j < layerSizes[1]; j++) {
                    bias1[j] -= learningRate * hidden1Gradients[j];
                }
            }

            if (epoch % 100 == 0) {
                std::println("EPOCH : {}  MSE : {}",
                    epoch, totalError / inputs.size());
            }
        }
    }
};

int main() {
    std::println("🧠 Neural Network (FNN) Implementation");
    try {
        NeuralNetwork nn(2, 8, 4, 1);
        std::mt19937 gen(std::random_device {}());
        std::uniform_real_distribution <> dist(-2.0, 2.0);
        const int numSamples = 1000;
        std::vector<std::vector<double>> inputs(numSamples);
        std::vector<std::vector<double>> targets(numSamples);

        for (int i = 0; i < numSamples; i++) {
            double x = dist(gen);
            double y = dist(gen);
            inputs[i] = {x, y};
            double distance = sqrt(x * x + y * y);
            targets[i] = {distance < 1.0 ? 1.0 : 0.0};
        }

        auto start = std::chrono::high_resolution_clock::now();
        nn.train(inputs, targets, 0.01, 1000);
        auto end = std::chrono::high_resolution_clock::now();
        std::println("Training Time : {} ms\n",
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

        std::vector<std::vector<double>> testPoints = {
            {0.0, 0.0},
            {0.1, 0.2},
            {-0.1, 0.2},
            {0.1, -0.2},
            {0.3, 0.4},
            {1.0, 1.0},
            {0.5, 0.5},
            {2.0, 0.0},
            {2.5, 0.0},
            {0.0, 2.5},
            {0.0, 2.0},
            {1.5, 1.5},
            {0.6, 0.5},
            {0.9, 0.1},
            {0.9, 0.9}
        };

        std::println("Enter a point (x, y) to test (or type 'exit' to quit):");
        std::vector<double> test_point(2);
        while (true) {
            std::string input;
            std::getline(std::cin, input);
            if (input == "x") {
                break;
            }

            try {
                size_t commaPos = input.find(',');
                if (commaPos == std::string::npos) {
                    throw std::invalid_argument("Invalid input format. Use 'x,y'.");
                }

                test_point[0] = std::stod(input.substr(0, commaPos));
                test_point[1] = std::stod(input.substr(commaPos + 1));

                auto output = nn.forward(test_point);
                double actual = sqrt(test_point[0] * test_point[0] + test_point[1] * test_point[1]) < 1.0 ? 1.0 : 0.0;
                std::println(" point ( {} , {} ) : {} ( actual : {} , error : {} )",
                    test_point[0], test_point[1], output[0], actual, abs(output[0] - actual));
            } catch (const std::exception &e) {
                std::println(" ERROR : {}", e.what());
            }
        }

        std::println("🧪 Testing Neural Network with Sample Points\n");

        for (std::size_t i = 0; i < testPoints.size(); ++i) {
            std::println(
                "Point {}: ({}, {})",
                i,
                testPoints[i][0],
                testPoints[i][1]);
        }

        std::println("🧑‍💻Test Results (1 = inside , 0 = outside) : \n");
        int count{0};
        int inside_count{0};
        for (const auto &point : testPoints) {
            auto output = nn.forward(point);
            double actual = sqrt(point[0] * point[0] + point[1] * point[1]) < 1.0 ? 1.0 : 0.0;
            std::println(" point ( {} , {} ) : {} ( actual : {} , error : {} )",
                point[0], point[1], output[0], actual, abs(output[0] - actual));
            if (abs(output[0] - actual) < 0.5) {
                count++;
            }
            if (actual == 1.0) {
                inside_count++;
            }
        }
        std::println("Correct Predictions : {}/{}", count, testPoints.size());
        std::println("Points Inside Circle : {}", inside_count);
    } catch(const std::runtime_error &e) {
        std::println(" ERROR : {}", e.what());
        return 1;
    }

    return 0;
}
