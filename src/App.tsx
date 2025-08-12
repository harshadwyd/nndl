import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter, Cell } from 'recharts';
import { Play, Pause, RotateCcw, Settings, Eye, EyeOff, Zap, Brain, TrendingUp, BarChart3, Activity, Target, Layers, Cpu, Database, BookOpen, Save, FolderOpen, X, Lightbulb, HelpCircle, RefreshCw } from 'lucide-react';

// --- Enhanced Matrix Math Utilities ---
// Tutorial: These are helper functions for matrix operations, which are like grids of numbers used in neural networks.
// Think of matrices as spreadsheets where each cell holds a number representing connections or data.
// Operations like 'dot' multiply matrices, which is how signals pass through the network.
const mat = {
  random: (rows, cols, scale = 1) => Array(rows).fill(0).map(() => Array(cols).fill(0).map(() => (Math.random() * 2 - 1) * scale)),
  zeros: (rows, cols) => Array(rows).fill(0).map(() => Array(cols).fill(0)),
  ones: (rows, cols) => Array(rows).fill(0).map(() => Array(cols).fill(1)),
  identity: (size) => Array(size).fill(0).map((_, i) => Array(size).fill(0).map((_, j) => i === j ? 1 : 0)),
  transpose: a => a[0].map((_, colIndex) => a.map(row => row[colIndex])),
  dot: (a, b) => {
    const aRows = a.length;
    const aCols = a[0].length;
    const bRows = b.length;
    const bCols = b[0].length;
    if (aCols !== bRows) throw new Error("Matrix dimensions are not compatible for dot product.");
    let result = mat.zeros(aRows, bCols);
    for (let i = 0; i < aRows; i++) {
        for (let j = 0; j < bCols; j++) {
            let sum = 0;
            for (let k = 0; k < aCols; k++) {
                sum += a[i][k] * b[k][j];
            }
            result[i][j] = sum;
        }
    }
    return result;
  },
  apply: (m, fn) => m.map((row, rowIndex) => row.map((val, colIndex) => fn(val, rowIndex, colIndex))),
  add: (a, b) => a.map((row, i) => row.map((val, j) => val + b[i][j])),
  subtract: (a, b) => a.map((row, i) => row.map((val, j) => val - b[i][j])),
  multiply: (a, b) => a.map((row, i) => row.map((val, j) => val * b[i][j])),
  scale: (m, s) => mat.apply(m, x => x * s),
  norm: (m) => Math.sqrt(m.reduce((sum, row) => sum + row.reduce((rowSum, val) => rowSum + val * val, 0), 0)),
  mean: (m) => m.reduce((sum, row) => sum + row.reduce((a, b) => a + b, 0), 0) / (m.length * m[0].length),
  max: (m) => Math.max(...m.flat()),
  min: (m) => Math.min(...m.flat()),
  xavierInit: (rows, cols) => mat.random(rows, cols, Math.sqrt(2 / (rows + cols))),
  heInit: (rows, cols) => mat.random(rows, cols, Math.sqrt(2 / rows)),
};

// --- Activation Functions ---
// Tutorial: Activation functions are like switches in neurons. They decide if a neuron "fires" based on input.
// Without them, the network would be too simple and couldn't learn complex things.
// The derivative helps during learning to adjust the network.
const activations = {
  sigmoid: { fn: z => 1.0 / (1.0 + Math.exp(-Math.max(-500, Math.min(500, z)))), derivative: z => { const sig = activations.sigmoid.fn(z); return sig * (1 - sig); } },
  relu: { fn: z => Math.max(0, z), derivative: z => z > 0 ? 1 : 0 },
  leakyRelu: { fn: z => z > 0 ? z : 0.01 * z, derivative: z => z > 0 ? 1 : 0.01 },
  tanh: { fn: z => Math.tanh(z), derivative: z => 1 - Math.pow(Math.tanh(z), 2) },
  swish: { fn: z => z / (1 + Math.exp(-z)), derivative: z => { const sig = activations.sigmoid.fn(z); return sig + z * sig * (1 - sig); } },
  linear: { fn: z => z, derivative: z => 1 }
};

// --- Loss Functions ---
// Tutorial: Loss is like a score of how wrong the network's guess is.
// Lower loss means better guesses. The derivative tells us how to improve.
const lossFunctions = {
  mse: { fn: (p, a) => { const d = mat.subtract(p, a); return mat.apply(d, x => x * x).reduce((s, r) => s + r.reduce((c, v) => c + v, 0), 0) / (2 * p.length); }, derivative: (p, a) => mat.subtract(p, a) },
  crossEntropy: { fn: (p, a) => { let l = 0; for (let i = 0; i < p.length; i++) for (let j = 0; j < p[i].length; j++) { const v = Math.max(1e-15, Math.min(1 - 1e-15, p[i][j])); l -= a[i][j] * Math.log(v) + (1 - a[i][j]) * Math.log(1 - v); } return l / p.length; }, derivative: (p, a) => p.map((r, i) => r.map((v, j) => { const val = Math.max(1e-15, Math.min(1 - 1e-15, v)); return (val - a[i][j]) / (val * (1 - val)); })) }
};

// --- Neural Network Implementation ---
// Tutorial: This is the brain! A neural network is like a team of workers (neurons) in layers.
// Input goes in one side, guesses come out the other. It learns by adjusting connections (weights).
class NeuralNetwork {
    constructor(layerSizes, activationFunctionName = 'sigmoid', initMethod = 'random', lossFunction = 'mse') {
        this.sizes = layerSizes;
        this.numLayers = layerSizes.length;
        this.activationFunctionName = activationFunctionName;
        this.lossFunction = lossFunction;
        this.initMethod = initMethod;
        this.initializeWeights();
        this.vWeights = this.weights.map(w => mat.zeros(w.length, w[0].length));
        this.vBiases = this.biases.map(b => mat.zeros(b.length, b[0].length));
        this.mWeights = this.weights.map(w => mat.zeros(w.length, w[0].length));
        this.mBiases = this.biases.map(b => mat.zeros(b.length, b[0].length));
        this.adamT = 0;
    }
    // Tutorial: Starting weights matter! Random starts are like guessing; better methods help learn faster.
    initializeWeights() {
        switch (this.initMethod) {
            case 'xavier': this.weights = this.sizes.slice(0, -1).map((x, i) => mat.xavierInit(this.sizes[i + 1], x)); break;
            case 'he': this.weights = this.sizes.slice(0, -1).map((x, i) => mat.heInit(this.sizes[i + 1], x)); break;
            case 'zeros': this.weights = this.sizes.slice(0, -1).map((x, i) => mat.zeros(this.sizes[i + 1], x)); break;
            default: this.weights = this.sizes.slice(0, -1).map((x, i) => mat.random(this.sizes[i + 1], x));
        }
        this.biases = this.sizes.slice(1).map(y => mat.zeros(y, 1));
    }
    _activation(z) { return activations[this.activationFunctionName].fn(z); }
    _activationPrime(z) { return activations[this.activationFunctionName].derivative(z); }
    // Tutorial: Forward pass is the network's "thinking" step. Input flows forward to make a guess.
    forwardPass(a, recordHistory = false) {
        let currentActivation = a;
        let activationRecord = recordHistory ? [a] : null;
        let zRecord = recordHistory ? [] : null;
        for (let i = 0; i < this.numLayers - 1; i++) {
            const z = mat.add(mat.dot(this.weights[i], currentActivation), this.biases[i]);
            if (recordHistory) zRecord.push(z);
            currentActivation = mat.apply(z, val => this._activation(val));
            if (recordHistory) activationRecord.push(currentActivation);
        }
        return recordHistory ? { output: currentActivation, activations: activationRecord, zs: zRecord } : currentActivation;
    }
    // Tutorial: Backpropagation is how the network learns! It figures out mistakes and fixes them backward.
    // Like tracing why a wrong answer happened in a chain of calculations.
    backpropagate(x, y) {
        let nabla_b = this.biases.map(b => mat.zeros(b.length, b[0].length));
        let nabla_w = this.weights.map(w => mat.zeros(w.length, w[0].length));
        const { activations, zs } = this.forwardPass(x, true);
        let delta = lossFunctions[this.lossFunction].derivative(activations[activations.length - 1], y);
        if (this.lossFunction === 'mse') {
            delta = mat.multiply(delta, mat.apply(zs[zs.length - 1], val => this._activationPrime(val)));
        }
        nabla_b[nabla_b.length - 1] = delta;
        nabla_w[nabla_w.length - 1] = mat.dot(delta, mat.transpose(activations[activations.length - 2]));
        for (let l = 2; l < this.numLayers; l++) {
            const sp_l = mat.apply(zs[zs.length - l], val => this._activationPrime(val));
            delta = mat.multiply(mat.dot(mat.transpose(this.weights[this.weights.length - l + 1]), delta), sp_l);
            nabla_b[nabla_b.length - l] = delta;
            nabla_w[nabla_w.length - l] = mat.dot(delta, mat.transpose(activations[activations.length - l - 1]));
        }
        return { nabla_b, nabla_w };
    }
    // Tutorial: After finding mistakes, update connections. Different ways (optimizers) make learning smoother.
    updateMiniBatch(miniBatch, learningRate, optimizer = 'sgd', momentum = 0.9, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8) {
        let nabla_b = this.biases.map(b => mat.zeros(b.length, b[0].length));
        let nabla_w = this.weights.map(w => mat.zeros(w.length, w[0].length));
        miniBatch.forEach(({ x, y }) => {
            const { nabla_b: dnb, nabla_w: dnw } = this.backpropagate(x, y);
            nabla_b = nabla_b.map((nb, i) => mat.add(nb, dnb[i]));
            nabla_w = nabla_w.map((nw, i) => mat.add(nw, dnw[i]));
        });
        const scale = 1.0 / miniBatch.length;
        nabla_b = nabla_b.map(nb => mat.scale(nb, scale));
        nabla_w = nabla_w.map(nw => mat.scale(nw, scale));
        switch (optimizer) {
            case 'momentum':
                this.vWeights = this.vWeights.map((v, i) => mat.add(mat.scale(v, momentum), mat.scale(nabla_w[i], learningRate)));
                this.vBiases = this.vBiases.map((v, i) => mat.add(mat.scale(v, momentum), mat.scale(nabla_b[i], learningRate)));
                this.weights = this.weights.map((w, i) => mat.subtract(w, this.vWeights[i]));
                this.biases = this.biases.map((b, i) => mat.subtract(b, this.vBiases[i]));
                break;
            case 'adam':
                this.adamT++;
                this.mWeights = this.mWeights.map((m, i) => mat.add(mat.scale(m, beta1), mat.scale(nabla_w[i], 1 - beta1)));
                this.mBiases = this.mBiases.map((m, i) => mat.add(mat.scale(m, beta1), mat.scale(nabla_b[i], 1 - beta1)));
                this.vWeights = this.vWeights.map((v, i) => mat.add(mat.scale(v, beta2), mat.scale(mat.apply(nabla_w[i], x => x * x), 1 - beta2)));
                this.vBiases = this.vBiases.map((v, i) => mat.add(mat.scale(v, beta2), mat.scale(mat.apply(nabla_b[i], x => x * x), 1 - beta2)));
                const mHatW = this.mWeights.map(m => mat.scale(m, 1 / (1 - Math.pow(beta1, this.adamT))));
                const mHatB = this.mBiases.map(m => mat.scale(m, 1 / (1 - Math.pow(beta1, this.adamT))));
                const vHatW = this.vWeights.map(v => mat.scale(v, 1 / (1 - Math.pow(beta2, this.adamT))));
                const vHatB = this.vBiases.map(v => mat.scale(v, 1 / (1 - Math.pow(beta2, this.adamT))));
                this.weights = this.weights.map((w, i) => mat.subtract(w, mat.apply(mHatW[i], (val, r, c) => learningRate * val / (Math.sqrt(vHatW[i][r][c]) + epsilon))));
                this.biases = this.biases.map((b, i) => mat.subtract(b, mat.apply(mHatB[i], (val, r, c) => learningRate * val / (Math.sqrt(vHatB[i][r][c]) + epsilon))));
                break;
            default:
                this.weights = this.weights.map((w, i) => mat.subtract(w, mat.scale(nabla_w[i], learningRate)));
                this.biases = this.biases.map((b, i) => mat.subtract(b, mat.scale(nabla_b[i], learningRate)));
        }
    }
    calculateLoss(data) { const lossFunc = lossFunctions[this.lossFunction]; let total = 0; data.forEach(({ x, y }) => { total += lossFunc.fn(this.forwardPass(x), y); }); return total / data.length; }
    calculateAccuracy(data) { let correct = 0; data.forEach(({ x, y }) => { const out = this.forwardPass(x); const pred = out[0][0] > 0.5 ? 1 : 0; if (pred === y[0][0]) correct++; }); return correct / data.length; }
    getWeightStats() { const w = this.weights.flat(2); if (w.length === 0) return { mean: 0, std: 0, min: 0, max: 0, norm: 0 }; const mean = w.reduce((a, b) => a + b, 0) / w.length; const std = Math.sqrt(w.reduce((s, v) => s + Math.pow(v - mean, 2), 0) / w.length); return { mean, std, min: Math.min(...w), max: Math.max(...w), norm: Math.sqrt(w.reduce((s, v) => s + v * v, 0)) }; }
}

// --- Datasets ---
// Tutorial: Datasets are like puzzles for the network to solve. XOR is tricky because it's not a straight line pattern.
const datasets = {
    XOR: { inputs: [[0, 0], [0, 1], [1, 0], [1, 1]], outputs: [[0], [1], [1], [0]], description: "XOR: Like a puzzle where same inputs give 0, different give 1." },
    AND: { inputs: [[0, 0], [0, 1], [1, 0], [1, 1]], outputs: [[0], [0], [0], [1]], description: "AND: Only true if both are true." },
    OR: { inputs: [[0, 0], [0, 1], [1, 0], [1, 1]], outputs: [[0], [1], [1], [1]], description: "OR: True if at least one is true." },
    CIRCLE: (() => { const i = [], o = []; for (let k = 0; k < 100; k++) { const x = Math.random() * 2 - 1, y = Math.random() * 2 - 1; i.push([x, y]); o.push([x * x + y * y < 0.5 ? 1 : 0]); } return { inputs: i, outputs: o, description: "CIRCLE: Points inside a circle are 1, outside are 0." }; })(),
    SPIRAL: (() => { const i = [], o = []; for (let k = 0; k < 200; k++) { const r = k/200*5, t = 1.75*k/200*2*Math.PI, x = r*Math.cos(t)*(k%2===0?1:-1), y = r*Math.sin(t)*(k%2===0?1:-1); i.push([x/5, y/5]); o.push([k%2]); } return { inputs: i, outputs: o, description: "SPIRAL: A swirling pattern to classify." }; })()
};
const formatData = name => { const d = datasets[name]; return d.inputs.map((input, i) => ({ x: input.map(v => [v]), y: d.outputs[i].map(v => [v]) })); };

// --- UI Components ---
// Tutorial: These are building blocks for the screen. Sliders let you change numbers easily.
const TooltipWrapper = ({ children, text }) => {
  const [show, setShow] = useState(false);
  return (
    <div className="relative inline-flex items-center" onMouseEnter={() => setShow(true)} onMouseLeave={() => setShow(false)}>
      {children}
      <HelpCircle className="ml-1 text-gray-400 cursor-help" size={14} />
      {show && <div className="absolute z-10 top-full left-0 bg-gray-700 p-2 rounded shadow-lg text-xs max-w-xs">{text}</div>}
    </div>
  );
};

const Slider = ({ label, value, min, max, step, onChange, help, disabled = false, tooltip }) => (
    <div className="mb-4">
        <label className="block text-sm font-medium text-gray-300 mb-1 flex items-center">
          {label}
          <TooltipWrapper text={tooltip}><span></span></TooltipWrapper>
        </label>
        <input
            type="range"
            min={min}
            max={max}
            step={step}
            value={value}
            onChange={e => onChange(parseFloat(e.target.value))}
            disabled={disabled}
            className="w-full h-2 bg-gray-600 rounded-lg appearance-none cursor-pointer disabled:opacity-50"
        />
        <div className="flex justify-between text-xs text-gray-400 mt-1">
            <span>{min}</span>
            <span className="font-bold text-cyan-400">{value}</span>
            <span>{max}</span>
        </div>
        {help && <p className="text-xs text-gray-500 mt-1">{help}</p>}
    </div>
);

const Select = ({ label, value, options, onChange, help, disabled = false, tooltip }) => (
    <div className="mb-4">
        <label className="block text-sm font-medium text-gray-300 mb-1 flex items-center">
          {label}
          <TooltipWrapper text={tooltip}><span></span></TooltipWrapper>
        </label>
        <select
            value={value}
            onChange={e => onChange(e.target.value)}
            disabled={disabled}
            className="w-full p-2 bg-gray-700 border border-gray-600 rounded-md focus:ring-cyan-500 focus:border-cyan-500 disabled:opacity-50"
        >
            {options.map(opt => typeof opt === 'string' ? 
                <option key={opt} value={opt}>{opt}</option> : 
                <option key={opt.value} value={opt.value}>{opt.label}</option>
            )}
        </select>
        {help && <p className="text-xs text-gray-500 mt-1">{help}</p>}
    </div>
);

const Toggle = ({ label, value, onChange, help, disabled = false, tooltip }) => (
    <div className="mb-4">
        <label className="flex items-center space-x-2">
            <input
                type="checkbox"
                checked={value}
                onChange={e => onChange(e.target.checked)}
                disabled={disabled}
                className="form-checkbox h-4 w-4 text-cyan-600 disabled:opacity-50"
            />
            <span className="text-sm font-medium text-gray-300 flex items-center">
              {label}
              <TooltipWrapper text={tooltip}><span></span></TooltipWrapper>
            </span>
        </label>
        {help && <p className="text-xs text-gray-500 mt-1">{help}</p>}
    </div>
);

const defaultParams = {
  dataset: 'OR', activation: 'sigmoid', learningRate: 0.8, epochs: 500,
  numHiddenLayers: 1, hiddenNeurons: [2], optimizer: 'sgd', momentum: 0.9,
  beta1: 0.9, beta2: 0.999, batchSize: 1, lossFunction: 'mse',
  initMethod: 'random', addNoise: false, animationSpeed: 20
};

const Sidebar = ({ params, setParams, isTraining, onExport, onImport, onApplyEasiestConfig, advancedMode, setAdvancedMode, onResetToDefaults }) => {
    const [showAdvanced, setShowAdvanced] = useState(false);
    const fileInputRef = useRef(null);

    const handleHiddenLayerCountChange = (count) => {
        const newHiddenNeurons = [...params.hiddenNeurons];
        while (newHiddenNeurons.length < count) newHiddenNeurons.push(4);
        setParams({ ...params, numHiddenLayers: count, hiddenNeurons: newHiddenNeurons.slice(0, count) });
    };

    const handleNeuronCountChange = (layerIndex, count) => {
        const newHiddenNeurons = [...params.hiddenNeurons];
        newHiddenNeurons[layerIndex] = count;
        setParams({ ...params, hiddenNeurons: newHiddenNeurons });
    };

    return (
        <div className="p-6 bg-gray-800/50 rounded-lg h-full overflow-y-auto">
            <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-bold text-white flex items-center">
                    <Settings className="mr-2" size={20} />
                    Controls
                </h2>
                <div className="flex space-x-2">
                    <button
                        onClick={onApplyEasiestConfig}
                        className="p-2 bg-yellow-600 hover:bg-yellow-700 rounded-lg transition-colors"
                        title="Easy Start"
                    >
                        <Lightbulb size={16} />
                    </button>
                    <button
                        onClick={onExport}
                        className="p-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
                        title="Save Settings"
                    >
                        <Save size={16} />
                    </button>
                    <button
                        onClick={() => fileInputRef.current?.click()}
                        className="p-2 bg-green-600 hover:bg-green-700 rounded-lg transition-colors"
                        title="Load Settings"
                    >
                        <FolderOpen size={16} />
                    </button>
                    <input
                        ref={fileInputRef}
                        type="file"
                        accept=".json"
                        onChange={onImport}
                        className="hidden"
                    />
                    <button
                        onClick={onResetToDefaults}
                        className="p-2 bg-red-600 hover:bg-red-700 rounded-lg transition-colors"
                        title="Reset to Defaults"
                    >
                        <RefreshCw size={16} />
                    </button>
                </div>
            </div>
            
            <Toggle 
                label="Advanced Mode" 
                value={advancedMode} 
                onChange={setAdvancedMode} 
                help="Unlock more options for experts!" 
                tooltip="Turn this on to see extra tools and settings. Great for when you're ready to experiment more."
            />
            
            <fieldset disabled={isTraining} className="space-y-6">
                {/* Dataset Configuration */}
                <div className="border-l-4 border-cyan-500 pl-4">
                    <h3 className="text-lg font-semibold text-cyan-400 mb-3 flex items-center">
                        <Database className="mr-2" size={18} />
                        Problem Setup
                    </h3>
                    <Select 
                        label="Dataset (Choose Puzzle)" 
                        value={params.dataset} 
                        options={Object.keys(datasets)} 
                        onChange={v => setParams({ ...params, dataset: v })} 
                        help={datasets[params.dataset]?.description}
                        tooltip="Pick a simple problem for the network to learn, like deciding if two things are the same or different."
                    />
                </div>

                {/* Network Architecture */}
                <div className="border-l-4 border-purple-500 pl-4">
                    <h3 className="text-lg font-semibold text-purple-400 mb-3 flex items-center">
                        <Layers className="mr-2" size={18} />
                        Network Architecture
                    </h3>
                    <Slider 
                        label="Hidden Layers (Teams of Workers)" 
                        value={params.numHiddenLayers} 
                        min={1} 
                        max={advancedMode ? 4 : 2} 
                        step={1} 
                        onChange={handleHiddenLayerCountChange} 
                        help="How many middle teams of workers?" 
                        tooltip="Layers are like teams. More layers can solve harder puzzles, but start small!"
                    />
                    {Array.from({ length: params.numHiddenLayers }).map((_, i) => (
                        <Slider 
                            key={i} 
                            label={`No. of Neurons (Team ${i + 1} Size)`} 
                            value={params.hiddenNeurons[i] || 4} 
                            min={2} 
                            max={advancedMode ? 32 : 8} 
                            step={1} 
                            onChange={v => handleNeuronCountChange(i, v)} 
                            tooltip="Number of workers in this team. More workers can think about more details."
                        />
                    ))}
                    <Select 
                        label="Activation Function (Thinking Style)" 
                        value={params.activation} 
                        options={Object.keys(activations)} 
                        onChange={v => setParams({ ...params, activation: v })} 
                        help="How workers decide to pass messages." 
                        tooltip="This is like how excited a worker gets. 'Sigmoid' is smooth and good for starters."
                    />
                    {advancedMode && (
                      <Select 
                          label="Weight Initialization (Starting Connections)" 
                          value={params.initMethod} 
                          options={[
                              { value: 'random', label: 'Random' },
                              { value: 'xavier', label: 'Smart Start (Xavier)' },
                              { value: 'he', label: 'Smart Start (He)' },
                              { value: 'zeros', label: 'All Zero (Try This Last)' }
                          ]} 
                          onChange={v => setParams({ ...params, initMethod: v })} 
                          help="How to set up initial connections." 
                          tooltip="Connections start random, but smart methods help learn faster."
                      />
                    )}
                </div>

                {/* Training Configuration */}
                <div className="border-l-4 border-green-500 pl-4">
                    <h3 className="text-lg font-semibold text-green-400 mb-3 flex items-center">
                        <TrendingUp className="mr-2" size={18} />
                        Training Parameters
                    </h3>
                    {advancedMode && (
                      <Select 
                          label="Optimizer (Learning Helper)" 
                          value={params.optimizer} 
                          options={[
                              { value: 'sgd', label: 'Basic (SGD)' },
                              { value: 'momentum', label: 'With Speed (Momentum)' },
                              { value: 'adam', label: 'Smart (Adam)' }
                          ]} 
                          onChange={v => setParams({ ...params, optimizer: v })} 
                          help="How the network improves its guesses." 
                          tooltip="Helpers make learning smoother. Start with Basic."
                      />
                    )}
                    <Slider 
                        label="Learning Rate (Learning Speed)" 
                        value={params.learningRate} 
                        min={0.001} 
                        max={advancedMode ? 3 : 1} 
                        step={0.001} 
                        onChange={v => setParams({ ...params, learningRate: v })} 
                        help="How big each improvement step is." 
                        tooltip="Too fast might skip good spots; too slow takes forever. Try 0.8."
                    />
                    {advancedMode && (
                      <Select 
                          label="Loss Function (Mistake Measure)" 
                          value={params.lossFunction} 
                          options={Object.keys(lossFunctions)} 
                          onChange={v => setParams({ ...params, lossFunction: v })} 
                          help="How to score mistakes." 
                          tooltip="MSE is simple for numbers; CrossEntropy for yes/no decisions."
                      />
                    )}
                    <Slider 
                        label="Epochs (Practice Rounds)" 
                        value={params.epochs} 
                        min={100} 
                        max={advancedMode ? 10000 : 1000} 
                        step={100} 
                        onChange={v => setParams({ ...params, epochs: v })} 
                        help="How many times to practice the whole puzzle." 
                        tooltip="More rounds = better learning, but watch for too many!"
                    />
                    {advancedMode && (
                      <Slider 
                          label="Batch Size (Group Size)" 
                          value={params.batchSize} 
                          min={1} 
                          max={32} 
                          step={1} 
                          onChange={v => setParams({ ...params, batchSize: v })} 
                          help="How many puzzle pieces at once." 
                          tooltip="Small groups learn detailed; big groups are faster but rougher."
                      />
                    )}
                </div>

                {/* Advanced Settings */}
                {advancedMode && (
                  <div className="border-l-4 border-yellow-500 pl-4">
                      <button
                          onClick={() => setShowAdvanced(!showAdvanced)}
                          className="flex items-center text-lg font-semibold text-yellow-400 mb-3 hover:text-yellow-300 transition-colors w-full text-left"
                      >
                          <Cpu className="mr-2" size={18} />
                          Advanced Optimizer Settings
                          {showAdvanced ? <EyeOff className="ml-auto" size={16} /> : <Eye className="ml-auto" size={16} />}
                      </button>
                      
                      {showAdvanced && (
                          <div className="space-y-4">
                              {params.optimizer === 'momentum' && (
                                  <Slider 
                                      label="Momentum (Speed Boost)" 
                                      value={params.momentum} 
                                      min={0.1} 
                                      max={0.99} 
                                      step={0.01} 
                                      onChange={v => setParams({ ...params, momentum: v })} 
                                      help="Keeps improvements moving." 
                                      tooltip="Like pushing a swing harder each time."
                                  />
                              )}
                              {params.optimizer === 'adam' && (
                                  <>
                                      <Slider 
                                          label="Beta 1 (Memory 1)" 
                                          value={params.beta1} 
                                          min={0.8} 
                                          max={0.99} 
                                          step={0.01} 
                                          onChange={v => setParams({ ...params, beta1: v })} 
                                          help="Remembers recent changes." 
                                          tooltip="Helps adjust speed based on past."
                                      />
                                      <Slider 
                                          label="Beta 2 (Memory 2)" 
                                          value={params.beta2} 
                                          min={0.9} 
                                          max={0.999} 
                                          step={0.001} 
                                          onChange={v => setParams({ ...params, beta2: v })} 
                                          help="Remembers big changes." 
                                          tooltip="Smooths out bumpy learning."
                                      />
                                  </>
                              )}
                              <Toggle 
                                  label="Noise (Add Wiggles)" 
                                  value={params.addNoise} 
                                  onChange={v => setParams({ ...params, addNoise: v })} 
                                  help="Small random shakes to avoid getting stuck." 
                                  tooltip="Like jiggling a key in a lock."
                              />
                              <Slider 
                                  label="Animation Speed (Show Speed)" 
                                  value={params.animationSpeed} 
                                  min={1} 
                                  max={50} 
                                  step={1} 
                                  onChange={v => setParams({ ...params, animationSpeed: v })} 
                                  help="How fast to show learning." 
                                  tooltip="Slow it down to watch carefully."
                              />
                          </div>
                      )}
                  </div>
                )}
            </fieldset>
        </div>
    );
};

// Tutorial: Shows the network like a diagram. Colors show how "excited" neurons are.
const NetworkVisualization = ({ network, trainingData, currentSample }) => {
    if (!network) return null;
    
    const layers = [network.sizes[0], ...network.sizes.slice(1)];
    
    return (
        <div className="bg-gray-800/50 p-4 rounded-lg">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                <Brain className="mr-2" />
                Network View
            </h3>
            <div className="flex justify-center items-start space-x-4 sm:space-x-8 overflow-x-auto p-4">
                {layers.map((layerSize, layerIndex) => (
                    <div key={layerIndex} className="flex flex-col items-center space-y-2 flex-shrink-0">
                        <div className="text-xs text-gray-400 font-semibold capitalize">
                            {layerIndex === 0 ? 'Input Layer (Start)' : layerIndex === layers.length - 1 ? 'Output Layer (Guess)' : `Hidden Layer ${layerIndex} (Middle ${layerIndex})`}
                        </div>
                        <div className="flex flex-col space-y-1">
                            {Array.from({ length: layerSize }).map((_, neuronIndex) => {
                                let activation = 0;
                                if (currentSample !== null && trainingData && trainingData[currentSample]) {
                                    const result = network.forwardPass(trainingData[currentSample].x, true);
                                    activation = result.activations[layerIndex]?.[neuronIndex]?.[0] || 0;
                                }
                                
                                const intensity = Math.min(Math.max(activation, 0), 1);
                                const color = `rgba(45, 212, 191, ${0.3 + intensity * 0.7})`;
                                
                                return (
                                    <div
                                        key={neuronIndex}
                                        className="w-8 h-8 rounded-full border-2 border-cyan-400 flex items-center justify-center text-xs font-bold text-white"
                                        style={{ backgroundColor: color }}
                                        title={`Activation (Excitement): ${activation.toFixed(3)}`}
                                    >
                                        {activation.toFixed(1)}
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

// Tutorial: Graph shows if the network is improving. Loss down = good!
const LossChart = ({ data }) => (
    <div className="bg-gray-800/50 p-4 rounded-lg" style={{ height: '400px' }}>
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
            <BarChart3 className="mr-2" />
            Learning Progress
        </h3>
        <ResponsiveContainer width="100%" height={320}>
            <LineChart data={data} margin={{ top: 5, right: 20, left: -10, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#4A5568" />
                <XAxis dataKey="epoch" stroke="#A0AEC0" />
                <YAxis yAxisId="loss" stroke="#2DD4BF" />
                <YAxis yAxisId="acc" orientation="right" stroke="#F59E0B" domain={[0, 1]} />
                <Tooltip 
                    contentStyle={{ backgroundColor: '#1A202C', border: '1px solid #4A5568' }}
                    formatter={(value, name) => [
                        typeof value === 'number' ? value.toFixed(4) : value,
                        name === 'loss' ? 'Loss (Mistake Score)' : 'Accuracy (Correct %)'
                    ]}
                />
                <Legend />
                <Line yAxisId="loss" type="monotone" dataKey="loss" stroke="#2DD4BF" strokeWidth={2} dot={false} name="Loss (Mistake Score)" />
                <Line yAxisId="acc" type="monotone" dataKey="accuracy" stroke="#F59E0B" strokeWidth={2} dot={false} name="Accuracy (Correct %)" />
            </LineChart>
        </ResponsiveContainer>
    </div>
);

// Tutorial: Table shows what the network guesses vs. what's right.
const PredictionsTable = ({ network, trainingData, currentSample }) => {
    if (!network || !trainingData || trainingData.length === 0) return null;
    
    return (
        <div className="bg-gray-800/50 p-4 rounded-lg h-80 overflow-y-auto">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                <Target className="mr-2" />
                Output Predictions
            </h3>
            <div className="overflow-x-auto">
                <table className="w-full text-left text-sm">
                    <thead>
                        <tr className="border-b border-gray-600">
                            <th className="p-2">Input</th>
                            <th className="p-2 text-center">Target</th>
                            <th className="p-2 text-center">Prediction (Network's Guess)</th>
                            <th className="p-2 text-center">Correct? (Good?)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {trainingData.map(({ x, y }, i) => {
                            const prediction = network.forwardPass(x);
                            const predicted = prediction[0][0];
                            const target = y[0][0];
                            const isCorrect = (predicted > 0.5 ? 1 : 0) === target;
                            const isCurrentSample = currentSample === i;
                            
                            return (
                                <tr 
                                    key={i} 
                                    className={`border-b border-gray-700 ${isCurrentSample ? 'bg-cyan-900/30' : 'hover:bg-gray-700/30'}`}
                                >
                                    <td className="p-2 font-mono text-xs">[{x.flat().map(v => v.toFixed(1)).join(', ')}]</td>
                                    <td className="p-2 text-center font-mono">{target}</td>
                                    <td className="p-2 text-center font-mono text-cyan-400">{predicted.toFixed(4)}</td>
                                    <td className="p-2 text-center">
                                        {isCorrect ? 
                                            <span className="text-green-400">✓</span> : 
                                            <span className="text-red-400">✗</span>
                                        }
                                    </td>
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

// Tutorial: Peek inside the network's "brain" - numbers that change as it learns.
const WeightsDisplay = ({ network }) => {
    if (!network) return null;
    
    const stats = network.getWeightStats();
    
    return (
        <div className="bg-gray-800/50 p-4 rounded-lg">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                <Activity className="mr-2" />
                Weight & Bias Inspector
            </h3>
            
            {/* Weight Statistics */}
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-4">
                <div className="bg-gray-900/50 p-3 rounded"><div className="text-xs text-gray-400">Mean (Average)</div><div className="font-mono text-sm">{stats.mean.toFixed(3)}</div></div>
                <div className="bg-gray-900/50 p-3 rounded"><div className="text-xs text-gray-400">Std Dev (Spread)</div><div className="font-mono text-sm">{stats.std.toFixed(3)}</div></div>
                <div className="bg-gray-900/50 p-3 rounded"><div className="text-xs text-gray-400">Min (Smallest)</div><div className="font-mono text-sm">{stats.min.toFixed(3)}</div></div>
                <div className="bg-gray-900/50 p-3 rounded"><div className="text-xs text-gray-400">Max (Biggest)</div><div className="font-mono text-sm">{stats.max.toFixed(3)}</div></div>
                <div className="bg-gray-900/50 p-3 rounded"><div className="text-xs text-gray-400">L2 Norm (Total Strength)</div><div className="font-mono text-sm">{stats.norm.toFixed(3)}</div></div>
            </div>
            
            {/* Detailed Weights and Biases */}
            <div className="space-y-4 max-h-96 overflow-y-auto">
                {network.weights.map((w, i) => (
                    <div key={i} className="border border-gray-600 rounded p-3">
                        <h4 className="font-semibold text-cyan-400 mb-2">Layer {i} to {i + 1}</h4>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                           <div>
                                <p className="text-sm font-medium text-gray-300 mb-1">Weights (Connections) ({w.length}×{w[0].length}):</p>
                                <div className="p-2 bg-gray-900 rounded-md font-mono text-xs overflow-x-auto max-h-32 overflow-y-auto">
                                    {w.map((row, r_idx) => ( <div key={r_idx} className="whitespace-nowrap">[{row.map(val => val.toFixed(2)).join(', ')}]</div> ))}
                                </div>
                           </div>
                           <div>
                                <p className="text-sm font-medium text-gray-300 mb-1">Biases (Boosts) ({network.biases[i].length}×1):</p>
                                <div className="p-2 bg-gray-900 rounded-md font-mono text-xs overflow-x-auto">
                                    {network.biases[i].map((row, r_idx) => ( <div key={r_idx}>[{row.map(val => val.toFixed(2)).join(', ')}]</div> ))}
                                </div>
                           </div>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

// Tutorial: Dot plot of the puzzle. Colors show groups to separate.
const DataVisualization = ({ trainingData, network }) => {
    if (!trainingData || trainingData.length === 0 || trainingData[0].x.length !== 2) return null;
    
    const plotData = trainingData.map(({ x, y }) => ({
        x: x[0][0],
        y: x[1][0],
        target: y[0][0],
    }));
    
    return (
        <div className="bg-gray-800/50 p-4 rounded-lg h-80">
            <h3 className="text-lg font-semibold text-white mb-4">Data Visualization</h3>
            <ResponsiveContainer width="100%" height="85%">
                <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#4A5568" />
                    <XAxis type="number" dataKey="x" name="X" domain={[-1.1, 1.1]} stroke="#A0AEC0" />
                    <YAxis type="number" dataKey="y" name="Y" domain={[-1.1, 1.1]} stroke="#A0AEC0" />
                    <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#1A202C', border: '1px solid #4A5568' }}/>
                    <Scatter name="Data Points" data={plotData} fill="#8884d8">
                       {plotData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.target === 1 ? '#2DD4BF' : '#F59E0B'} />
                        ))}
                    </Scatter>
                </ScatterChart>
            </ResponsiveContainer>
        </div>
    );
};

// Tutorial: Quick stats on how training is going.
const TrainingStats = ({ network, trainingData, epoch, totalEpochs }) => {
    if (!network || !trainingData || trainingData.length === 0) return null;
    
    const loss = network.calculateLoss(trainingData);
    const accuracy = network.calculateAccuracy(trainingData);
    const progress = totalEpochs > 0 ? (epoch / totalEpochs) * 100 : 0;
    
    return (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="bg-gray-800/50 p-4 rounded-lg">
                <div className="flex items-center text-cyan-400 mb-2"><TrendingUp className="mr-2" size={16} /><span className="text-sm font-medium">Progress</span></div>
                <div className="text-2xl font-bold text-white">{progress.toFixed(1)}%</div>
                <div className="w-full bg-gray-600 rounded-full h-2 mt-2"><div className="bg-cyan-500 h-2 rounded-full transition-all duration-300" style={{ width: `${progress}%` }}></div></div>
            </div>
            <div className="bg-gray-800/50 p-4 rounded-lg">
                <div className="flex items-center text-red-400 mb-2"><Activity className="mr-2" size={16} /><span className="text-sm font-medium">Loss (Mistake Score)</span></div>
                <div className="text-2xl font-bold text-white">{loss.toFixed(4)}</div>
                <div className="text-xs text-gray-400">(Lower is better)</div>
            </div>
            <div className="bg-gray-800/50 p-4 rounded-lg">
                <div className="flex items-center text-green-400 mb-2"><Target className="mr-2" size={16} /><span className="text-sm font-medium">Accuracy (Correct %)</span></div>
                <div className="text-2xl font-bold text-white">{(accuracy * 100).toFixed(1)}%</div>
                <div className="text-xs text-gray-400">How many right</div>
            </div>
            <div className="bg-gray-800/50 p-4 rounded-lg">
                <div className="flex items-center text-purple-400 mb-2"><Zap className="mr-2" size={16} /><span className="text-sm font-medium">Epoch (Round)</span></div>
                <div className="text-2xl font-bold text-white">{epoch}</div>
                <div className="text-xs text-gray-400">of {totalEpochs}</div>
            </div>
        </div>
    );
};

// --- Concepts ---
// Tutorial: Simple explanations with fun analogies.
const concepts = {
  feedforward: { title: 'Feedforward (🧠 Thinking Forward)', analogy: 'Passing a note in class.', explanation: 'The network takes info and passes it through teams to make a guess.', inSimulator: 'Watch circles light up left to right in Network View.' },
  backpropagation: { title: 'Backpropagation (💡 Learning from Mistakes)', analogy: 'Fixing a wrong answer step by step.', explanation: 'Checks guess, traces back errors, tweaks connections to improve.', inSimulator: 'Mistake Score drops, Correct % rises in charts.' },
  neuron: { title: 'Neuron (⚪ Worker)', analogy: 'A tiny decision maker.', explanation: 'Takes messages, thinks, sends new message.', inSimulator: 'Each circle in Network View is one.' },
  weights: { title: 'Weights & Biases (🎛️ Connections & Boosts)', analogy: 'Strength of friendships.', explanation: 'Numbers showing how much one worker listens to another. Learns by changing these.', inSimulator: 'See numbers in Inside the Network.' },
  epoch: { title: 'Epoch (🔄 Practice Round)', analogy: 'One full game.', explanation: 'Goes through all puzzle pieces once.', inSimulator: 'Round counter in stats.' }
};

const ConceptCorner = React.forwardRef((props, ref) => {
  const [activeConcept, setActiveConcept] = useState('feedforward');
  return (
    <div ref={ref} className="bg-gray-800/50 p-6 rounded-lg scroll-mt-6">
      <h3 className="text-lg font-semibold text-white mb-4 flex items-center"><Lightbulb className="mr-2 text-yellow-400"/>Key Concepts</h3>
      <div className="flex flex-col md:flex-row gap-6">
        <div className="flex flex-row md:flex-col gap-2 flex-wrap md:flex-nowrap">
          {Object.keys(concepts).map(key => (
            <button key={key} onClick={() => setActiveConcept(key)} className={`p-2 rounded-md text-left text-sm font-semibold transition-colors w-full flex items-center gap-2 ${activeConcept === key ? 'bg-cyan-600 text-white' : 'bg-gray-700 hover:bg-gray-600'}`}>
              {concepts[key].title}
            </button>
          ))}
        </div>
        <div className="flex-1 bg-gray-900/50 p-4 rounded-md">
          <p className="text-gray-400 italic mb-2">"{concepts[activeConcept].analogy}"</p>
          <p className="text-gray-200 mb-4">{concepts[activeConcept].explanation}</p>
          <p className="text-cyan-300 bg-cyan-900/30 p-3 rounded-md border-l-4 border-cyan-400"><strong className="font-bold">In this Playground:</strong> {concepts[activeConcept].inSimulator}</p>
        </div>
      </div>
    </div>
  );
});

// --- Onboarding and Guided Exercises ---
const Onboarding = ({ onDismiss }) => {
    const [step, setStep] = useState(0);

    const steps = [
        {
            title: "Welcome to the Neural Network Playground!",
            text: "Neural networks are like smart teams that learn from examples. They make a guess, check if they're wrong, and then improve for the next time.",
            hint: "Click 'Next' to see how it works."
        },
        {
            title: "Your Goal",
            text: "You'll build a small neural network (a team of workers), give it a puzzle (a dataset), and watch it learn to solve it!",
            hint: "Use the controls on the left to change the setup."
        },
        {
            title: "First Exercise: The 'OR' Puzzle",
            text: "Let's start easy. Use the default settings: Dataset = 'OR', 1 Hidden Layer with 2 Neurons. Set Learning Rate to 0.8 and Epochs to 500.",
            hint: "Press the 'Start Learning' button and watch the 'Loss (Mistake Score)' go down!"
        },
        {
            title: "Observe the Learning Process",
            text: "You can see the network 'thinking' in the Network View, see its progress in the charts, and check its specific guesses in the table.",
            hint: "If the network gets stuck, just hit 'Restart' and try again."
        },
        {
            title: "Now, It's Your Turn!",
            text: "Try changing the settings. What happens if you add more layers or neurons? What if you increase the learning speed?",
            hint: "Don't be afraid to experiment! You can always reset to the default settings."
        }
    ];

    const currentStep = steps[step];

    return (
        <div className="fixed inset-0 bg-black/70 z-50 flex items-center justify-center p-4">
            <div className="bg-gray-800 border-2 border-cyan-500 rounded-lg shadow-2xl max-w-md w-full p-6 relative animate-fade-in">
                <button onClick={onDismiss} className="absolute top-2 right-2 text-gray-400 hover:text-white"><X size={20}/></button>
                <h2 className="text-xl font-bold text-cyan-400 mb-2">{currentStep.title}</h2>
                <p className="text-gray-300 mb-4">{currentStep.text}</p>
                <p className="text-yellow-300 bg-gray-900/50 p-2 rounded-md mb-4"><strong>Hint:</strong> {currentStep.hint}</p>
                <div className="flex justify-between items-center">
                    <button 
                        onClick={onDismiss}
                        className="px-3 py-2 bg-gray-600 hover:bg-gray-700 rounded-md font-semibold text-sm"
                    >
                        Skip Tutorial
                    </button>
                    <div>
                        <span className="text-sm text-gray-500 mr-4">Step {step + 1} of {steps.length}</span>
                        <button 
                            onClick={() => step < steps.length - 1 ? setStep(s => s + 1) : onDismiss()}
                            className="px-4 py-2 bg-cyan-600 hover:bg-cyan-700 rounded-md font-semibold"
                        >
                            {step < steps.length - 1 ? 'Next' : 'Start Exploring!'}
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

// --- Main App ---
export default function App() {
    const [params, setParams] = useState(defaultParams);
    const [network, setNetwork] = useState(null);
    const [lossHistory, setLossHistory] = useState([]);
    const [isTraining, setIsTraining] = useState(false);
    const [currentEpoch, setCurrentEpoch] = useState(0);
    const [currentSample, setCurrentSample] = useState(0);
    const [showWeights, setShowWeights] = useState(false);
    const [isPaused, setIsPaused] = useState(false);
    const [showOnboarding, setShowOnboarding] = useState(true);
    const [advancedMode, setAdvancedMode] = useState(false);
    const conceptsRef = useRef(null);

    const trainingData = useMemo(() => formatData(params.dataset), [params.dataset]);
    
    const layerSizes = useMemo(() => {
        if (!trainingData || trainingData.length === 0) return [];
        const inputSize = trainingData[0].x.length;
        const outputSize = trainingData[0].y.length;
        return [inputSize, ...params.hiddenNeurons.slice(0, params.numHiddenLayers), outputSize];
    }, [trainingData, params.hiddenNeurons, params.numHiddenLayers]);

    const initializeNetwork = useCallback(() => {
        if (layerSizes.length === 0) return;
        const newNet = new NeuralNetwork(layerSizes, params.activation, params.initMethod, params.lossFunction);
        setNetwork(newNet);
        const initialLoss = newNet.calculateLoss(trainingData);
        const initialAccuracy = newNet.calculateAccuracy(trainingData);
        setLossHistory([{ epoch: 0, loss: initialLoss, accuracy: initialAccuracy }]);
        setCurrentEpoch(0);
        setCurrentSample(0);
    }, [layerSizes, params.activation, params.initMethod, params.lossFunction, trainingData]);
    
    useEffect(() => {
        if (!isTraining) {
            initializeNetwork();
        }
    }, [params, initializeNetwork]);

    useEffect(() => {
        let timeoutId;
        if (isTraining && network && currentEpoch < params.epochs && !isPaused) {
            const trainStep = () => {
                const shuffledData = [...trainingData].sort(() => Math.random() - 0.5);
                const miniBatch = shuffledData.slice(0, params.batchSize);
                if (miniBatch.length > 0) {
                    network.updateMiniBatch(miniBatch, params.learningRate, params.optimizer, params.momentum, params.beta1, params.beta2);
                }
                setCurrentSample((prev) => (prev + 1) % trainingData.length);
                if (currentEpoch % Math.max(1, Math.floor(params.epochs / 200)) === 0 || currentEpoch === params.epochs - 1) {
                    const currentLoss = network.calculateLoss(trainingData);
                    const currentAccuracy = network.calculateAccuracy(trainingData);
                    setLossHistory(prev => [...prev, { epoch: currentEpoch + 1, loss: currentLoss, accuracy: currentAccuracy }]);
                    setNetwork(net => Object.assign(Object.create(Object.getPrototypeOf(net)), net));
                }
                setCurrentEpoch(prev => prev + 1);
            };
            timeoutId = setTimeout(trainStep, Math.max(1, 100 - params.animationSpeed * 2));
        } else if (isTraining && currentEpoch >= params.epochs) {
            setIsTraining(false);
            setIsPaused(false);
        }
        return () => clearTimeout(timeoutId);
    }, [isTraining, isPaused, network, currentEpoch, params, trainingData]);

    const isFinished = currentEpoch >= params.epochs;

    const handleStartPause = () => {
        if (!isTraining || isFinished) {
            if (showOnboarding) setShowOnboarding(false);
        }

        if (isFinished) {
            handleReset();
            setTimeout(() => setIsTraining(true), 100);
        } else if (isTraining) {
            setIsPaused(!isPaused);
        } else {
            setIsTraining(true);
            setIsPaused(false);
        }
    };

    const handleReset = () => {
        setIsTraining(false);
        setIsPaused(false);
        initializeNetwork();
    };
    
    const handleExport = () => {
        const exportData = { params, lossHistory, currentEpoch };
        const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `network_settings_${params.dataset}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    };

    const handleImport = (event) => {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                try {
                    const importData = JSON.parse(e.target.result);
                    setParams(importData.params);
                    setLossHistory(importData.lossHistory || []);
                    setCurrentEpoch(importData.currentEpoch || 0);
                    handleReset();
                } catch (error) {
                    alert('Oops! Could not load that file.');
                }
            };
            reader.readAsText(file);
        }
    };

    const handleScrollToConcepts = () => {
        conceptsRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    const handleApplyEasiestConfig = () => {
        setParams(defaultParams);
    };

    const handleResetToDefaults = () => {
        setParams(defaultParams);
        setAdvancedMode(false);
        handleReset();
    };

    return (
        <div className="bg-gray-900 text-white min-h-screen font-sans p-4 sm:p-6 lg:p-8">
            {showOnboarding && <Onboarding onDismiss={() => setShowOnboarding(false)} />}
            <div className="max-w-7xl mx-auto">
                <header className="text-center mb-8">
                    <h1 className="text-4xl font-bold text-white mb-2 flex items-center justify-center">
                        <Brain className="mr-3 text-cyan-400" size={40} />
                        Neural Network Playground
                    </h1>
                    <p className="text-gray-400 max-w-3xl mx-auto">
                        Build and train your own neural network! Watch it learn to solve different puzzles step-by-step.
                    </p>
                </header>

                <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
                    <div className="lg:col-span-1">
                        <Sidebar 
                            params={params} 
                            setParams={setParams} 
                            isTraining={isTraining}
                            onExport={handleExport}
                            onImport={handleImport}
                            onApplyEasiestConfig={handleApplyEasiestConfig}
                            advancedMode={advancedMode}
                            setAdvancedMode={setAdvancedMode}
                            onResetToDefaults={handleResetToDefaults}
                        />
                    </div>

                    <main className="lg:col-span-3 space-y-6">
                        <div className="flex flex-wrap gap-4">
                            <button onClick={handleStartPause} className="flex-1 min-w-40 flex items-center justify-center p-3 bg-cyan-600 hover:bg-cyan-700 rounded-lg font-semibold transition-colors disabled:bg-gray-500">
                                {isFinished ? <RotateCcw className="mr-2" /> : (isTraining && !isPaused ? <Pause className="mr-2" /> : <Play className="mr-2" />)}
                                {isFinished ? 'Start New Training' : (isTraining ? (isPaused ? 'Resume Training' : 'Pause Training') : 'Start Training')}
                            </button>
                            <button onClick={handleReset} disabled={!isTraining && currentEpoch === 0} className="flex-1 min-w-40 flex items-center justify-center p-3 bg-gray-600 hover:bg-gray-700 rounded-lg font-semibold transition-colors disabled:opacity-50">
                                <RotateCcw className="mr-2" />
                                Reset
                            </button>
                            <button onClick={() => setShowWeights(!showWeights)} className="flex items-center justify-center p-3 bg-purple-600 hover:bg-purple-700 rounded-lg font-semibold transition-colors">
                                {showWeights ? <EyeOff className="mr-2" /> : <Eye className="mr-2" />}
                                {showWeights ? 'Hide Inspector' : 'Show Inspector'}
                            </button>
                            <button onClick={handleScrollToConcepts} className="flex items-center justify-center p-3 bg-yellow-600 hover:bg-yellow-700 rounded-lg font-semibold transition-colors">
                                <BookOpen className="mr-2" />
                                Key Concepts
                            </button>
                        </div>
                        
                        <TrainingStats 
                            network={network} 
                            trainingData={trainingData} 
                            epoch={currentEpoch} 
                            totalEpochs={params.epochs} 
                        />
                        
                        <NetworkVisualization 
                            network={network} 
                            trainingData={trainingData} 
                            currentSample={currentSample} 
                        />
                        
                        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
                            <LossChart data={lossHistory} />
                            <PredictionsTable 
                                network={network} 
                                trainingData={trainingData} 
                                currentSample={currentSample} 
                            />
                        </div>
                        
                        {trainingData[0]?.x.length === 2 && (
                            <DataVisualization 
                                trainingData={trainingData} 
                                network={network} 
                            />
                        )}

                        {showWeights && <WeightsDisplay network={network} />}
                        
                        <ConceptCorner ref={conceptsRef} />
                    </main>
                </div>
            </div>
        </div>
    );
}