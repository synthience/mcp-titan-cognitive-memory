import * as tf from '@tensorflow/tfjs';

// Basic interface for an in-house "tensor" object
export interface ITensor extends tf.TensorContainerObject {
  dataSync(): number[];
  dispose(): void;
  shape: number[];
  [key: string]: any; // For TensorContainerObject
}

export interface ITensorOps {
  tensor(data: number[], shape?: number[]): ITensor;
  tensor1d(data: number[]): ITensor;
  scalar(value: number): ITensor;
  zeros(shape: number[]): ITensor;
  randomNormal(shape: number[]): ITensor;
  variable(tensor: ITensor): ITensor;
  tidy<T extends tf.TensorContainer>(fn: () => T): T;
  train: {
    adam: (learningRate: number) => {
      minimize: (lossFn: () => tf.Scalar) => ITensor;
    };
  };
  concat(tensors: ITensor[], axis?: number): ITensor;
  matMul(a: ITensor, b: ITensor): ITensor;
  sub(a: ITensor, b: ITensor): ITensor;
  add(a: ITensor, b: ITensor): ITensor;
  mul(a: ITensor, b: ITensor): ITensor;
  div(a: ITensor, b: ITensor): ITensor;
  relu(x: ITensor): ITensor;
  sigmoid(x: ITensor): ITensor;
  tanh(x: ITensor): ITensor;
  mean(x: ITensor, axis?: number): ITensor;
  sum(x: ITensor, axis?: number): ITensor;
  sqrt(x: ITensor): ITensor;
  exp(x: ITensor): ITensor;
  log(x: ITensor): ITensor;
  dispose(): void;
  memory(): { numTensors: number; numDataBuffers: number; numBytes: number };
}

export interface IMemoryModel {
  forward(x: ITensor, memoryState: ITensor): {
    predicted: ITensor;
    newMemory: ITensor;
    surprise: ITensor;
  };
  trainStep(x_t: ITensor, x_next: ITensor, memoryState: ITensor): ITensor;
  manifoldStep(base: ITensor, velocity: ITensor): ITensor;
  saveModel(path: string): Promise<void>;
  loadModel(path: string): Promise<void>;
  getConfig(): any;
}

// Simple wrapper
export class TensorWrapper implements ITensor {
  constructor(private tensor: tf.Tensor) {}

  [key: string]: any; // Required for TensorContainerObject

  static fromTensor(tensor: tf.Tensor): TensorWrapper {
    return new TensorWrapper(tensor);
  }

  get shape(): number[] {
    return this.tensor.shape;
  }

  dataSync(): number[] {
    return Array.from(this.tensor.dataSync());
  }

  dispose(): void {
    this.tensor.dispose();
  }

  toJSON(): any {
    return {
      dataSync: this.dataSync(),
      shape: this.shape
    };
  }
}

export function wrapTensor(tensor: tf.Tensor): ITensor {
  return TensorWrapper.fromTensor(tensor);
}

export function unwrapTensor(tensor: ITensor): tf.Tensor {
  if (tensor instanceof TensorWrapper) {
    return (tensor as any).tensor;
  }
  throw new Error('Cannot unwrap non-TensorWrapper object');
}
