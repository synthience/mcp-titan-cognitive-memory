import * as tf from '@tensorflow/tfjs';
export interface ITensor extends tf.TensorContainerObject {
    dataSync(): number[];
    dispose(): void;
    shape: number[];
    [key: string]: any;
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
    memory(): {
        numTensors: number;
        numDataBuffers: number;
        numBytes: number;
    };
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
export declare class TensorWrapper implements ITensor {
    private tensor;
    constructor(tensor: tf.Tensor);
    [key: string]: any;
    static fromTensor(tensor: tf.Tensor): TensorWrapper;
    get shape(): number[];
    dataSync(): number[];
    dispose(): void;
    toJSON(): any;
}
export declare function wrapTensor(tensor: tf.Tensor): ITensor;
export declare function unwrapTensor(tensor: ITensor): tf.Tensor;
