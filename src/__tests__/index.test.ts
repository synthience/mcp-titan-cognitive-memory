import * as tf from '@tensorflow/tfjs';
import { TitanMemoryModel } from '../model.js';
import { wrapTensor } from '../types.js';

describe('TitanMemoryModel Tests', () => {
  let model: TitanMemoryModel;
  const inputDim = 64;
  const hiddenDim = 32;
  const outputDim = 64;

  beforeEach(() => {
    model = new TitanMemoryModel({
      inputDim,
      hiddenDim,
      outputDim,
      learningRate: 0.001
    });
  });

  test('Model processes sequences correctly', () => {
    const x = wrapTensor(tf.randomNormal([inputDim]));
    const memoryState = wrapTensor(tf.zeros([outputDim]));
    
    const { predicted, newMemory, surprise } = model.forward(x, memoryState);
    
    // Expect the shapes to match the logic:
    // predicted => [inputDim]
    // newMemory => [outputDim]
    expect(predicted.shape).toEqual([inputDim]);
    expect(newMemory.shape).toEqual([outputDim]);
    // surprise => scalar
    expect(surprise.shape).toEqual([]);
    
    predicted.dispose();
    newMemory.dispose();
    surprise.dispose();
    x.dispose();
    memoryState.dispose();
  });

  test('Training reduces loss over time', () => {
    const memoryState = wrapTensor(tf.zeros([outputDim]));
    const x_t = wrapTensor(tf.randomNormal([inputDim]));
    const x_next = wrapTensor(tf.randomNormal([inputDim]));
    
    // Run multiple training steps
    const losses: number[] = [];
    for (let i = 0; i < 10; i++) {
      const loss = model.trainStep(x_t, x_next, memoryState);
      losses.push(loss.dataSync()[0]);
      loss.dispose();
    }
    
    // Check if loss generally decreases
    const firstLoss = losses[0];
    const lastLoss = losses[losses.length - 1];
    expect(lastLoss).toBeLessThan(firstLoss);
    
    x_t.dispose();
    x_next.dispose();
    memoryState.dispose();
  });

  test('Manifold updates work correctly when enabled', () => {
    model = new TitanMemoryModel({
      inputDim,
      hiddenDim,
      outputDim,
      useManifold: true
    });

    const base = wrapTensor(tf.randomNormal([inputDim]));
    const velocity = wrapTensor(tf.randomNormal([inputDim]));
    
    const result = model.manifoldStep(base, velocity);
    
    // We check that the result is normalized (approximately),
    // or at least consistent with the manifold logic.
    const resultTensor = tf.tensor(result.dataSync());
    const normVal = resultTensor.norm().dataSync()[0];
    // For a sphere manifold, we expect roughly norm ~ 1.0
    expect(Math.abs(normVal - 1.0)).toBeLessThan(1e-5);
    
    base.dispose();
    velocity.dispose();
    result.dispose();
    resultTensor.dispose();
  });

  test('Model can save and load weights', async () => {
    const x = wrapTensor(tf.randomNormal([inputDim]));
    const memoryState = wrapTensor(tf.zeros([outputDim]));
    
    // Get initial prediction
    const { predicted: pred1 } = model.forward(x, memoryState);
    const initial = pred1.dataSync();
    pred1.dispose();
    
    // Save and load weights
    await model.saveModel('./test-weights.json');
    await model.loadModel('./test-weights.json');
    
    // Get prediction after loading
    const { predicted: pred2 } = model.forward(x, memoryState);
    const loaded = pred2.dataSync();
    pred2.dispose();
    
    // Compare predictions
    expect(Array.from(initial)).toEqual(Array.from(loaded));
    
    x.dispose();
    memoryState.dispose();
  });
});
