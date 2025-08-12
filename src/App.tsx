import React, { useState, useEffect, useCallback } from 'react';
import { Brain, Play, RotateCcw, Info, X, CheckCircle, XCircle, AlertCircle } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

// Neural Network Implementation
class NeuralNetwork {
  private weights1: number[][];
  private weights2: number[][];
  private bias1: number[];
  private bias2: number[];
  private learningRate: number;
  private trainingHistory: Array<{ epoch: number; loss: number; accuracy: number }> = [];

  constructor(inputSize: number, hiddenSize: number, outputSize: number, learningRate: number = 0.1) {
    this.learningRate = learningRate;
    
    // Initialize weights with Xavier initialization
    this.weights1 = this.initializeWeights(inputSize, hiddenSize);
    this.weights2 = this.initializeWeights(hiddenSize, outputSize);
    this.bias1 = new Array(hiddenSize).fill(0);
    this.bias2 = new Array(outputSize).fill(0);
  }

  private initializeWeights(rows: number, cols: number): number[][] {
    const weights = [];
    const limit = Math.sqrt(6 / (rows + cols));
    for (let i = 0; i < rows; i++) {
      weights[i] = [];
      for (let j = 0; j < cols; j++) {
        weights[i][j] = (Math.random() * 2 - 1) * limit;
      }
    }
    return weights;
  }

  private sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x))));
  }

  private sigmoidDerivative(x: number): number {
    return x * (1 - x);
  }

  private softmax(arr: number[]): number[] {
    const max = Math.max(...arr);
    const exp = arr.map(x => Math.exp(x - max));
    const sum = exp.reduce((a, b) => a + b, 0);
    return exp.map(x => x / sum);
  }

  forward(inputs: number[]): { hidden: number[]; output: number[] } {
    // Hidden layer
    const hidden = this.weights1[0].map((_, j) => {
      let sum = this.bias1[j];
      for (let i = 0; i < inputs.length; i++) {
        sum += inputs[i] * this.weights1[i][j];
      }
      return this.sigmoid(sum);
    });

    // Output layer
    const outputRaw = this.weights2[0].map((_, j) => {
      let sum = this.bias2[j];
      for (let i = 0; i < hidden.length; i++) {
        sum += hidden[i] * this.weights2[i][j];
      }
      return sum;
    });

    const output = this.softmax(outputRaw);
    return { hidden, output };
  }

  train(inputs: number[], targets: number[]): number {
    const { hidden, output } = this.forward(inputs);

    // Calculate output layer errors
    const outputErrors = output.map((o, i) => targets[i] - o);
    
    // Calculate hidden layer errors
    const hiddenErrors = hidden.map((_, i) => {
      let error = 0;
      for (let j = 0; j < output.length; j++) {
        error += outputErrors[j] * this.weights2[i][j];
      }
      return error * this.sigmoidDerivative(hidden[i]);
    });

    // Update weights and biases
    for (let i = 0; i < this.weights2.length; i++) {
      for (let j = 0; j < this.weights2[i].length; j++) {
        this.weights2[i][j] += this.learningRate * outputErrors[j] * hidden[i];
      }
    }

    for (let j = 0; j < this.bias2.length; j++) {
      this.bias2[j] += this.learningRate * outputErrors[j];
    }

    for (let i = 0; i < this.weights1.length; i++) {
      for (let j = 0; j < this.weights1[i].length; j++) {
        this.weights1[i][j] += this.learningRate * hiddenErrors[j] * inputs[i];
      }
    }

    for (let j = 0; j < this.bias1.length; j++) {
      this.bias1[j] += this.learningRate * hiddenErrors[j];
    }

    // Calculate loss (mean squared error)
    const loss = outputErrors.reduce((sum, error) => sum + error * error, 0) / outputErrors.length;
    return loss;
  }

  predict(inputs: number[]): number[] {
    return this.forward(inputs).output;
  }

  addTrainingPoint(epoch: number, loss: number, accuracy: number) {
    this.trainingHistory.push({ epoch, loss, accuracy });
  }

  getTrainingHistory() {
    return this.trainingHistory;
  }

  reset() {
    const inputSize = this.weights1.length;
    const hiddenSize = this.weights1[0].length;
    const outputSize = this.weights2[0].length;
    
    this.weights1 = this.initializeWeights(inputSize, hiddenSize);
    this.weights2 = this.initializeWeights(hiddenSize, outputSize);
    this.bias1 = new Array(hiddenSize).fill(0);
    this.bias2 = new Array(outputSize).fill(0);
    this.trainingHistory = [];
  }
}

// Training data for different problems
const PROBLEMS = {
  XOR: {
    name: 'XOR Gate',
    description: 'Learn the XOR logical operation',
    inputs: [[0, 0], [0, 1], [1, 0], [1, 1]],
    outputs: [[1, 0], [0, 1], [0, 1], [1, 0]], // [False, True] format
    inputLabels: ['Input A', 'Input B'],
    outputLabels: ['False', 'True'],
    expectedResults: [
      { input: [0, 0], expected: 'False', explanation: '0 XOR 0 = False' },
      { input: [0, 1], expected: 'True', explanation: '0 XOR 1 = True' },
      { input: [1, 0], expected: 'True', explanation: '1 XOR 0 = True' },
      { input: [1, 1], expected: 'False', explanation: '1 XOR 1 = False' }
    ]
  },
  AND: {
    name: 'AND Gate',
    description: 'Learn the AND logical operation',
    inputs: [[0, 0], [0, 1], [1, 0], [1, 1]],
    outputs: [[1, 0], [1, 0], [1, 0], [0, 1]], // [False, True] format
    inputLabels: ['Input A', 'Input B'],
    outputLabels: ['False', 'True'],
    expectedResults: [
      { input: [0, 0], expected: 'False', explanation: '0 AND 0 = False' },
      { input: [0, 1], expected: 'False', explanation: '0 AND 1 = False' },
      { input: [1, 0], expected: 'False', explanation: '1 AND 0 = False' },
      { input: [1, 1], expected: 'True', explanation: '1 AND 1 = True' }
    ]
  },
  OR: {
    name: 'OR Gate',
    description: 'Learn the OR logical operation',
    inputs: [[0, 0], [0, 1], [1, 0], [1, 1]],
    outputs: [[1, 0], [0, 1], [0, 1], [0, 1]], // [False, True] format
    inputLabels: ['Input A', 'Input B'],
    outputLabels: ['False', 'True'],
    expectedResults: [
      { input: [0, 0], expected: 'False', explanation: '0 OR 0 = False' },
      { input: [0, 1], expected: 'True', explanation: '0 OR 1 = True' },
      { input: [1, 0], expected: 'True', explanation: '1 OR 0 = True' },
      { input: [1, 1], expected: 'True', explanation: '1 OR 1 = True' }
    ]
  }
};

interface ComparisonResult {
  input: number[];
  expected: string;
  predicted: string;
  confidence: number;
  isCorrect: boolean;
  explanation: string;
}

function App() {
  const [network, setNetwork] = useState<NeuralNetwork | null>(null);
  const [selectedProblem, setSelectedProblem] = useState<keyof typeof PROBLEMS>('OR');
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [showOnboarding, setShowOnboarding] = useState(true);
  const [showResults, setShowResults] = useState(false);
  const [comparisonResults, setComparisonResults] = useState<ComparisonResult[]>([]);
  const [trainingComplete, setTrainingComplete] = useState(false);

  // Initialize network
  useEffect(() => {
    const nn = new NeuralNetwork(2, 4, 2, 0.5);
    setNetwork(nn);
  }, []);

  const roundToDecimalPlaces = (num: number, places: number = 2): number => {
    return Math.round(num * Math.pow(10, places)) / Math.pow(10, places);
  };

  const getPredictedLabel = (output: number[]): { label: string; confidence: number } => {
    const maxIndex = output.indexOf(Math.max(...output));
    const confidence = roundToDecimalPlaces(output[maxIndex] * 100, 1);
    return {
      label: PROBLEMS[selectedProblem].outputLabels[maxIndex],
      confidence
    };
  };

  const testNetwork = useCallback(() => {
    if (!network) return;

    const problem = PROBLEMS[selectedProblem];
    const results: ComparisonResult[] = [];

    problem.expectedResults.forEach((testCase, index) => {
      const prediction = network.predict(testCase.input);
      const { label: predictedLabel, confidence } = getPredictedLabel(prediction);
      
      results.push({
        input: testCase.input,
        expected: testCase.expected,
        predicted: predictedLabel,
        confidence,
        isCorrect: predictedLabel === testCase.expected,
        explanation: testCase.explanation
      });
    });

    setComparisonResults(results);
    setShowResults(true);
  }, [network, selectedProblem]);

  const trainNetwork = async () => {
    if (!network) return;

    setIsTraining(true);
    setTrainingProgress(0);
    setTrainingComplete(false);
    
    const problem = PROBLEMS[selectedProblem];
    const epochs = 1000;

    for (let epoch = 0; epoch < epochs; epoch++) {
      let totalLoss = 0;
      let correctPredictions = 0;

      // Train on all examples
      for (let i = 0; i < problem.inputs.length; i++) {
        const loss = network.train(problem.inputs[i], problem.outputs[i]);
        totalLoss += loss;

        // Check accuracy
        const prediction = network.predict(problem.inputs[i]);
        const predictedClass = prediction.indexOf(Math.max(...prediction));
        const actualClass = problem.outputs[i].indexOf(Math.max(...problem.outputs[i]));
        if (predictedClass === actualClass) correctPredictions++;
      }

      const avgLoss = totalLoss / problem.inputs.length;
      const accuracy = (correctPredictions / problem.inputs.length) * 100;
      
      if (epoch % 10 === 0) {
        network.addTrainingPoint(epoch, avgLoss, accuracy);
      }

      setTrainingProgress((epoch / epochs) * 100);

      // Add small delay for visual feedback
      if (epoch % 50 === 0) {
        await new Promise(resolve => setTimeout(resolve, 10));
      }
    }

    setIsTraining(false);
    setTrainingComplete(true);
    testNetwork();
  };

  const resetNetwork = () => {
    if (network) {
      network.reset();
      setTrainingProgress(0);
      setShowResults(false);
      setComparisonResults([]);
      setTrainingComplete(false);
    }
  };

  const handleProblemChange = (problem: keyof typeof PROBLEMS) => {
    setSelectedProblem(problem);
    resetNetwork();
  };

  const closeOnboarding = () => {
    setShowOnboarding(false);
  };

  const currentProblem = PROBLEMS[selectedProblem];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      {/* Onboarding Modal */}
      {showOnboarding && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6">
              <div className="flex justify-between items-center mb-6">
                <div className="flex items-center space-x-3">
                  <Brain className="w-8 h-8 text-indigo-600" />
                  <h2 className="text-2xl font-bold text-gray-800">Welcome to Neural Network Playground</h2>
                </div>
                <button
                  onClick={closeOnboarding}
                  className="text-gray-400 hover:text-gray-600 transition-colors"
                >
                  <X className="w-6 h-6" />
                </button>
              </div>

              <div className="space-y-6">
                <div>
                  <h3 className="text-lg font-semibold text-gray-800 mb-3">What You'll Learn</h3>
                  <p className="text-gray-600 mb-4">
                    This playground demonstrates how neural networks learn logical operations. 
                    You can train networks to understand XOR, AND, and OR gates.
                  </p>
                </div>

                <div className="bg-indigo-50 rounded-lg p-4">
                  <h4 className="font-semibold text-indigo-800 mb-3">Default Example: OR Gate</h4>
                  <p className="text-indigo-700 mb-3">
                    The OR gate returns True if at least one input is True. Here's what the network should learn:
                  </p>
                  <div className="grid grid-cols-2 gap-3">
                    {currentProblem.expectedResults.map((result, index) => (
                      <div key={index} className="bg-white rounded-lg p-3 border border-indigo-200">
                        <div className="text-sm font-medium text-gray-800">
                          Input: [{result.input.join(', ')}]
                        </div>
                        <div className="text-sm text-indigo-600">
                          Expected: {result.expected}
                        </div>
                        <div className="text-xs text-gray-500 mt-1">
                          {result.explanation}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="bg-yellow-50 rounded-lg p-4">
                  <h4 className="font-semibold text-yellow-800 mb-2">How It Works</h4>
                  <ul className="text-yellow-700 text-sm space-y-1">
                    <li>• The network starts with random weights</li>
                    <li>• During training, it learns from examples</li>
                    <li>• After training, we compare predicted vs expected results</li>
                    <li>• Confidence scores show how certain the network is</li>
                  </ul>
                </div>

                <button
                  onClick={closeOnboarding}
                  className="w-full bg-indigo-600 text-white py-3 px-4 rounded-lg hover:bg-indigo-700 transition-colors font-medium"
                >
                  Start Learning!
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center space-x-3 mb-4">
            <Brain className="w-10 h-10 text-indigo-600" />
            <h1 className="text-4xl font-bold text-gray-800">Neural Network Playground</h1>
          </div>
          <p className="text-gray-600 text-lg">
            Train a neural network to learn logical operations and see how it compares to expected results
          </p>
        </div>

        {/* Problem Selection */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
          <h2 className="text-xl font-semibold text-gray-800 mb-4">Choose a Problem</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {Object.entries(PROBLEMS).map(([key, problem]) => (
              <button
                key={key}
                onClick={() => handleProblemChange(key as keyof typeof PROBLEMS)}
                className={`p-4 rounded-lg border-2 transition-all ${
                  selectedProblem === key
                    ? 'border-indigo-500 bg-indigo-50 text-indigo-700'
                    : 'border-gray-200 hover:border-gray-300 text-gray-700'
                }`}
              >
                <h3 className="font-semibold mb-2">{problem.name}</h3>
                <p className="text-sm opacity-75">{problem.description}</p>
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Training Controls */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Training Controls</h2>
            
            <div className="space-y-4">
              <div className="flex space-x-3">
                <button
                  onClick={trainNetwork}
                  disabled={isTraining}
                  className="flex-1 bg-indigo-600 text-white py-3 px-4 rounded-lg hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center space-x-2"
                >
                  <Play className="w-5 h-5" />
                  <span>{isTraining ? 'Training...' : 'Train Network'}</span>
                </button>
                
                <button
                  onClick={resetNetwork}
                  className="bg-gray-500 text-white py-3 px-4 rounded-lg hover:bg-gray-600 transition-colors flex items-center justify-center"
                >
                  <RotateCcw className="w-5 h-5" />
                </button>

                <button
                  onClick={() => setShowOnboarding(true)}
                  className="bg-blue-500 text-white py-3 px-4 rounded-lg hover:bg-blue-600 transition-colors flex items-center justify-center"
                >
                  <Info className="w-5 h-5" />
                </button>
              </div>

              {isTraining && (
                <div className="space-y-2">
                  <div className="flex justify-between text-sm text-gray-600">
                    <span>Training Progress</span>
                    <span>{roundToDecimalPlaces(trainingProgress, 1)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-indigo-600 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${trainingProgress}%` }}
                    />
                  </div>
                </div>
              )}

              {trainingComplete && (
                <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                  <div className="flex items-center space-x-2">
                    <CheckCircle className="w-5 h-5 text-green-600" />
                    <span className="text-green-800 font-medium">Training Complete!</span>
                  </div>
                  <p className="text-green-700 text-sm mt-1">
                    Network has been trained on {currentProblem.inputs.length} examples.
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Results Comparison */}
          {showResults && (
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-xl font-semibold text-gray-800 mb-4">Prediction Results</h2>
              
              <div className="space-y-3">
                {comparisonResults.map((result, index) => (
                  <div
                    key={index}
                    className={`border-2 rounded-lg p-4 ${
                      result.isCorrect
                        ? 'border-green-200 bg-green-50'
                        : 'border-red-200 bg-red-50'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        {result.isCorrect ? (
                          <CheckCircle className="w-5 h-5 text-green-600" />
                        ) : (
                          <XCircle className="w-5 h-5 text-red-600" />
                        )}
                        <span className="font-medium text-gray-800">
                          Input: [{result.input.join(', ')}]
                        </span>
                      </div>
                      <div className="flex items-center space-x-1">
                        <AlertCircle className="w-4 h-4 text-gray-500" />
                        <span className="text-sm text-gray-600">
                          {result.confidence}% confident
                        </span>
                      </div>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="text-gray-600">Expected: </span>
                        <span className="font-medium text-gray-800">{result.expected}</span>
                      </div>
                      <div>
                        <span className="text-gray-600">Predicted: </span>
                        <span className={`font-medium ${
                          result.isCorrect ? 'text-green-700' : 'text-red-700'
                        }`}>
                          {result.predicted}
                        </span>
                      </div>
                    </div>
                    
                    <div className="mt-2 text-xs text-gray-500 italic">
                      {result.explanation}
                    </div>
                  </div>
                ))}
              </div>

              <div className="mt-4 p-3 bg-gray-50 rounded-lg">
                <div className="text-sm text-gray-600">
                  <strong>Accuracy: </strong>
                  {roundToDecimalPlaces(
                    (comparisonResults.filter(r => r.isCorrect).length / comparisonResults.length) * 100,
                    1
                  )}% 
                  ({comparisonResults.filter(r => r.isCorrect).length}/{comparisonResults.length} correct)
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Training History Chart */}
        {network && network.getTrainingHistory().length > 0 && (
          <div className="bg-white rounded-xl shadow-lg p-6 mt-6">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Training History</h2>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={network.getTrainingHistory()}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="epoch" />
                  <YAxis yAxisId="left" />
                  <YAxis yAxisId="right" orientation="right" />
                  <Tooltip />
                  <Legend />
                  <Line
                    yAxisId="left"
                    type="monotone"
                    dataKey="loss"
                    stroke="#ef4444"
                    strokeWidth={2}
                    name="Loss"
                  />
                  <Line
                    yAxisId="right"
                    type="monotone"
                    dataKey="accuracy"
                    stroke="#10b981"
                    strokeWidth={2}
                    name="Accuracy (%)"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;