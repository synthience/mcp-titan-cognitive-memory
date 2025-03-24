import * as tf from '@tensorflow/tfjs';
import { ITensor, IMemoryModel, TensorWrapper, wrapTensor, unwrapTensor } from './types.js';
import * as fs from 'fs/promises';

export interface TitanMemoryConfig {
  inputDim?: number;
  hiddenDim?: number;
  outputDim?: number; // We'll treat this as "memoryDim" internally
  learningRate?: number;
  useManifold?: boolean;
  momentumFactor?: number;
  forgetGateInit?: number;
  maxStepSize?: number;
  tangentEpsilon?: number;
  numHeads?: number; // Number of attention heads
  numLayers?: number; // Number of hierarchical memory layers
}

interface ForwardResult extends tf.TensorContainerObject {
  predicted: ITensor;
  newMemory: ITensor;
  surprise: ITensor;
  [key: string]: any;
}

export class TitanMemoryModel implements IMemoryModel {
  private inputDim: number;
  private hiddenDim: number;
  private memoryDim: number;
  private learningRate: number;
  public useManifold: boolean;
  private momentumFactor: number;
  private forgetGateInit: number;
  private maxStepSize: number;
  private tangentEpsilon: number;
  private numHeads: number;
  private numLayers: number;

  // For convenience, total output dimension = memoryDim + inputDim
  private fullOutputDim: number;

  // Trainable parameters
  private W1: tf.Variable;
  private b1: tf.Variable;
  private W2: tf.Variable;
  private b2: tf.Variable;
  private forgetGate: tf.Variable;
  private optimizer: tf.Optimizer;

  // Attention parameters
  private queryWeights: tf.Variable[];
  private keyWeights: tf.Variable[];
  private valueWeights: tf.Variable[];
  private attentionOutputWeights: tf.Variable[];

  // Hierarchical memory
  private hierarchicalMemory: tf.Variable[];

  constructor(config: TitanMemoryConfig = {}) {
    this.inputDim = config.inputDim || 64;
    this.hiddenDim = config.hiddenDim || 32;

    // We interpret 'outputDim' from config as the memory dimension:
    this.memoryDim = config.outputDim || 64;

    this.fullOutputDim = this.inputDim + this.memoryDim;

    this.learningRate = config.learningRate || 1e-3;
    this.useManifold = config.useManifold || false;
    this.momentumFactor = config.momentumFactor || 0.9;
    this.forgetGateInit = config.forgetGateInit || 0.01;
    this.maxStepSize = config.maxStepSize || 0.1;
    this.tangentEpsilon = config.tangentEpsilon || 1e-8;
    this.numHeads = config.numHeads || 4;
    this.numLayers = config.numLayers || 3;

    // Initialize trainable parameters
    // First layer receives inputDim + memoryDim:
    this.W1 = tf.variable(tf.randomNormal([this.hiddenDim, this.inputDim + this.memoryDim], 0, 0.1));
    this.b1 = tf.variable(tf.zeros([this.hiddenDim]));

    // Second layer outputs (memoryDim + inputDim):
    this.W2 = tf.variable(tf.randomNormal([this.fullOutputDim, this.hiddenDim], 0, 0.1));
    this.b2 = tf.variable(tf.zeros([this.fullOutputDim]));

    // Forget gate
    this.forgetGate = tf.variable(tf.scalar(this.forgetGateInit));

    // Initialize optimizer
    this.optimizer = tf.train.adam(this.learningRate);

    // Initialize attention parameters
    this.queryWeights = [];
    this.keyWeights = [];
    this.valueWeights = [];
    this.attentionOutputWeights = [];
    for (let i = 0; i < this.numHeads; i++) {
      this.queryWeights.push(tf.variable(tf.randomNormal([this.memoryDim, this.memoryDim], 0, 0.1)));
      this.keyWeights.push(tf.variable(tf.randomNormal([this.memoryDim, this.memoryDim], 0, 0.1)));
      this.valueWeights.push(tf.variable(tf.randomNormal([this.memoryDim, this.memoryDim], 0, 0.1)));
      this.attentionOutputWeights.push(tf.variable(tf.randomNormal([this.memoryDim, this.memoryDim], 0, 0.1)));
    }

    // Initialize hierarchical memory
    this.hierarchicalMemory = [];
    for (let i = 0; i < this.numLayers; i++) {
      this.hierarchicalMemory.push(tf.variable(tf.zeros([this.memoryDim])));
    }
  }

  private multiHeadAttention(query: tf.Tensor, key: tf.Tensor, value: tf.Tensor): tf.Tensor {
    const attentionHeads = [];
    for (let i = 0; i < this.numHeads; i++) {
      const q = tf.matMul(query, this.queryWeights[i]);
      const k = tf.matMul(key, this.keyWeights[i]);
      const v = tf.matMul(value, this.valueWeights[i]);

      const attentionScores = tf.softmax(tf.matMul(q, k.transpose()).div(tf.scalar(Math.sqrt(this.memoryDim))));
      const attentionOutput = tf.matMul(attentionScores, v);

      attentionHeads.push(attentionOutput);
    }

    const concatenatedHeads = tf.concat(attentionHeads, -1);
    const output = tf.matMul(concatenatedHeads, this.attentionOutputWeights[0]);

    return output;
  }

  public forward(xTensor: ITensor, memoryState: ITensor): ForwardResult {
    const x = unwrapTensor(xTensor);       // shape [inputDim]
    const memory = unwrapTensor(memoryState); // shape [memoryDim]

    // Gate the memory state
    const forgetVal = this.forgetGate; // shape []
    const one = tf.scalar(1.0);
    const gatedMemory = tf.mul(memory, tf.sub(one, forgetVal)); // shape [memoryDim]

    // Combine input and gated memory => shape [inputDim + memoryDim]
    const combined = tf.concat([x, gatedMemory], 0);
    const combinedReshaped = combined.reshape([1, this.inputDim + this.memoryDim]);

    // MLP forward pass
    const hidden1 = tf.add(
      tf.matMul(combinedReshaped, this.W1.transpose()),
      this.b1
    ).relu(); // shape [1, hiddenDim]

    let out = tf.add(
      tf.matMul(hidden1, this.W2.transpose()),
      this.b2
    ).squeeze(); 

    // If out was completely scalar (which it shouldn't be), fix shape:
    if (out.shape.length === 0) {
      out = out.reshape([1]);
    }

    // Split output into new memory [0: memoryDim], predicted [memoryDim: memoryDim+inputDim]
    const newMemory = out.slice([0], [this.memoryDim]);
    const predicted = out.slice([this.memoryDim], [this.inputDim]);

    // Calculate surprise (MSE between predicted and x)
    const diff = tf.sub(predicted, x);
    const surprise = tf.mean(tf.square(diff)); // scalar

    // Multi-head attention for memory update
    const attentionOutput = this.multiHeadAttention(newMemory, memory, memory);

    // Hierarchical memory update
    for (let i = 0; i < this.numLayers; i++) {
      const layerMemory = this.hierarchicalMemory[i];
      const updatedLayerMemory = tf.add(layerMemory, attentionOutput);
      this.hierarchicalMemory[i].assign(updatedLayerMemory);
    }

    // Clean up intermediate tensors
    one.dispose();
    gatedMemory.dispose();
    combined.dispose();
    combinedReshaped.dispose();
    hidden1.dispose();
    out.dispose();
    diff.dispose();
    attentionOutput.dispose();

    return {
      predicted: wrapTensor(predicted),
      newMemory: wrapTensor(newMemory),
      surprise: wrapTensor(surprise)
    };
  }

  public manifoldStep(base: ITensor, velocity: ITensor): ITensor {
    // Riemannian "update" if useManifold is true
    if (!this.useManifold) {
      // Standard Euclidean update
      return wrapTensor(tf.add(unwrapTensor(base), unwrapTensor(velocity)));
    }

    const result = tf.tidy<ITensor>(() => {
      const baseTensor = unwrapTensor(base);
      const velocityTensor = unwrapTensor(velocity);

      const dot = baseTensor.mul(velocityTensor).sum(); // shape []
      const radial = baseTensor.mul(dot);               // shape [inputDim]
      const tangent = velocityTensor.sub(radial);       // shape [inputDim]
      const tnorm = tangent.norm();                     // shape []

      const tNormVal = tnorm.dataSync()[0];
      if (tNormVal < this.tangentEpsilon) {
        // Very small velocity => no movement
        return wrapTensor(baseTensor);
      }

      const stepSize = Math.min(tNormVal, this.maxStepSize);
      const direction = tangent.div(tf.scalar(tNormVal));
      const cosV = tf.cos(tf.scalar(stepSize));
      const sinV = tf.sin(tf.scalar(stepSize));
      const part1 = baseTensor.mul(cosV);
      const part2 = direction.mul(sinV);
      const newParam = part1.add(part2);
      const newParamNorm = newParam.norm();

      // Return a normalized param
      return wrapTensor(newParam.div(newParamNorm.add(tf.scalar(1e-12))));
    });

    return result;
  }

  public trainStep(x_t: ITensor, x_next: ITensor, memoryState: ITensor): ITensor {
    const xt = unwrapTensor(x_t);
    const xn = unwrapTensor(x_next);
    const mem = unwrapTensor(memoryState);

    // Minimizing a combined loss
    const cost = this.optimizer.minimize(() => {
      const { predicted, newMemory, surprise } = this.forward(x_t, memoryState);

      // MSE wrt x_next plus a small "surprise" penalty
      const diff = tf.sub(unwrapTensor(predicted), xn);
      const mse = tf.mean(tf.square(diff)).asScalar();
      const spr = unwrapTensor(surprise);
      
      // Clean up intermediate tensors
      diff.dispose();
      predicted.dispose();
      newMemory.dispose();

      // Weighted combination
      const totalLoss = tf.add(mse, tf.mul(tf.scalar(0.01), spr)).asScalar();
      
      // Clean up more tensors
      mse.dispose();
      spr.dispose();
      
      return totalLoss;
    }, true);

    // If cost is null for some reason, return scalar(0)
    const result = wrapTensor(cost || tf.scalar(0));

    return result;
  }

  public async saveModel(path: string): Promise<void> {
    const weights = {
      W1: await this.W1.array(),
      b1: await this.b1.array(),
      W2: await this.W2.array(),
      b2: await this.b2.array(),
      forgetGate: await this.forgetGate.array(),
      queryWeights: await Promise.all(this.queryWeights.map(w => w.array())),
      keyWeights: await Promise.all(this.keyWeights.map(w => w.array())),
      valueWeights: await Promise.all(this.valueWeights.map(w => w.array())),
      attentionOutputWeights: await Promise.all(this.attentionOutputWeights.map(w => w.array())),
      hierarchicalMemory: await Promise.all(this.hierarchicalMemory.map(m => m.array()))
    };

    await fs.writeFile(path.replace('file://',''), JSON.stringify(weights));
  }

  public async loadModel(path: string): Promise<void> {
    const weightsJson = await fs.readFile(path.replace('file://',''), 'utf8');
    const weights = JSON.parse(weightsJson);

    this.W1.assign(tf.tensor2d(weights.W1));
    this.b1.assign(tf.tensor1d(weights.b1));
    this.W2.assign(tf.tensor2d(weights.W2));
    this.b2.assign(tf.tensor1d(weights.b2));
    this.forgetGate.assign(tf.scalar(weights.forgetGate));
    this.queryWeights.forEach((w, i) => w.assign(tf.tensor2d(weights.queryWeights[i])));
    this.keyWeights.forEach((w, i) => w.assign(tf.tensor2d(weights.keyWeights[i])));
    this.valueWeights.forEach((w, i) => w.assign(tf.tensor2d(weights.valueWeights[i])));
    this.attentionOutputWeights.forEach((w, i) => w.assign(tf.tensor2d(weights.attentionOutputWeights[i])));
    this.hierarchicalMemory.forEach((m, i) => m.assign(tf.tensor1d(weights.hierarchicalMemory[i])));
  }

  public getConfig(): TitanMemoryConfig {
    return {
      inputDim: this.inputDim,
      hiddenDim: this.hiddenDim,
      outputDim: this.memoryDim, // We keep "outputDim" referring to memoryDim
      learningRate: this.learningRate,
      useManifold: this.useManifold,
      momentumFactor: this.momentumFactor,
      forgetGateInit: this.forgetGateInit,
      maxStepSize: this.maxStepSize,
      tangentEpsilon: this.tangentEpsilon,
      numHeads: this.numHeads,
      numLayers: this.numLayers
    };
  }

  public getWeights() {
    return {
      W1: this.W1.arraySync(),
      b1: this.b1.arraySync(),
      W2: this.W2.arraySync(),
      b2: this.b2.arraySync(),
      forgetGate: this.forgetGate.arraySync(),
      queryWeights: this.queryWeights.map(w => w.arraySync()),
      keyWeights: this.keyWeights.map(w => w.arraySync()),
      valueWeights: this.valueWeights.map(w => w.arraySync()),
      attentionOutputWeights: this.attentionOutputWeights.map(w => w.arraySync()),
      hierarchicalMemory: this.hierarchicalMemory.map(m => m.arraySync())
    };
  }
}
