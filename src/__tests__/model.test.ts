import * as tf from '@tensorflow/tfjs';
import { TitanMemoryModel } from '../model.js';
import { ITensor, wrapTensor, unwrapTensor } from '../types.js';

// Set backend to CPU for deterministic tests
tf.setBackend('cpu');
tf.env().set('WEBGL_FORCE_F16_TEXTURES', false);

describe('TitanMemoryModel', () => {
  let model: TitanMemoryModel;
  const inputDim = 64;
  const hiddenDim = 32;
  const outputDim = 64;

  beforeEach(() => {
    // Create model with default settings
    model = new TitanMemoryModel({
      inputDim,
      hiddenDim,
      outputDim,
      learningRate: 0.001
    });
  });

  afterEach(() => {
    // Clean up any remaining tensors
    tf.disposeVariables();
    tf.dispose(); // Clean up all tensors
  });

  test('initializes with correct dimensions', () => {
    const config = model.getConfig();
    expect(config.inputDim).toBe(inputDim);
    expect(config.hiddenDim).toBe(hiddenDim);
    expect(config.outputDim).toBe(outputDim);
  });

  test('forward pass produces correct output shapes', () => {
    const x = wrapTensor(tf.randomNormal([inputDim], 0, 1, 'float32'));
    const memoryState = wrapTensor(tf.zeros([outputDim]));
    
    const { predicted, newMemory, surprise } = model.forward(x, memoryState);
    
    expect(predicted.shape).toEqual([inputDim]);
    expect(newMemory.shape).toEqual([outputDim]);
    expect(surprise.shape).toEqual([]);

    // Clean up
    x.dispose();
    memoryState.dispose();
    predicted.dispose();
    newMemory.dispose();
    surprise.dispose();
  });

  describe('training', () => {
    test('reduces loss over time with default learning rate', () => {
      // Create input tensor and its target (same tensor)
      const x_t = tf.randomNormal([inputDim]);
      const x_next = x_t.clone();
      const memoryState = tf.zeros([outputDim]);

      // Wrap tensors for model
      const wrappedX = wrapTensor(x_t);
      const wrappedNext = wrapTensor(x_next);
      const wrappedMemory = wrapTensor(memoryState);
      
      const losses: number[] = [];
      const surprises: number[] = [];
      const numSteps = 50;
      
      for (let i = 0; i < numSteps; i++) {
        const cost = model.trainStep(wrappedX, wrappedNext, wrappedMemory);
        const { surprise } = model.forward(wrappedX, wrappedMemory);
        
        losses.push(unwrapTensor(cost).dataSync()[0]);
        surprises.push(unwrapTensor(surprise).dataSync()[0]);
        
        cost.dispose();
        surprise.dispose();
      }
      
      // Verify loss reduction
      const firstLosses = losses.slice(0, 5);
      const lastLosses = losses.slice(-5);
      const avgFirstLoss = firstLosses.reduce((a, b) => a + b, 0) / firstLosses.length;
      const avgLastLoss = lastLosses.reduce((a, b) => a + b, 0) / lastLosses.length;
      
      expect(avgLastLoss).toBeLessThan(avgFirstLoss);

      // Verify surprise reduction
      const firstSurprises = surprises.slice(0, 5);
      const lastSurprises = surprises.slice(-5);
      const avgFirstSurprise = firstSurprises.reduce((a, b) => a + b, 0) / firstSurprises.length;
      const avgLastSurprise = lastSurprises.reduce((a, b) => a + b, 0) / lastSurprises.length;
      
      expect(avgLastSurprise).toBeLessThan(avgFirstSurprise);

      // Clean up
      x_t.dispose();
      x_next.dispose();
      memoryState.dispose();
      wrappedX.dispose();
      wrappedNext.dispose();
      wrappedMemory.dispose();
    });

    test('trains with different learning rates', () => {
      const learningRates = [0.0001, 0.001, 0.01];
      const numSteps = 20;

      for (const lr of learningRates) {
        const testModel = new TitanMemoryModel({
          inputDim,
          hiddenDim,
          outputDim,
          learningRate: lr
        });

        const x_t = tf.randomNormal([inputDim]);
        const x_next = x_t.clone();
        const memoryState = tf.zeros([outputDim]);
        const wrappedX = wrapTensor(x_t);
        const wrappedNext = wrapTensor(x_next);
        const wrappedMemory = wrapTensor(memoryState);

        const losses: number[] = [];
        
        for (let i = 0; i < numSteps; i++) {
          const cost = testModel.trainStep(wrappedX, wrappedNext, wrappedMemory);
          losses.push(unwrapTensor(cost).dataSync()[0]);
          cost.dispose();
        }

        const avgFirstLoss = losses.slice(0, 3).reduce((a, b) => a + b, 0) / 3;
        const avgLastLoss = losses.slice(-3).reduce((a, b) => a + b, 0) / 3;
        
        expect(avgLastLoss).toBeLessThan(avgFirstLoss);

        // Clean up
        x_t.dispose();
        x_next.dispose();
        memoryState.dispose();
        wrappedX.dispose();
        wrappedNext.dispose();
        wrappedMemory.dispose();
      }
    });

    test('handles sequence training', () => {
      return tf.tidy(() => {
        const sequenceLength = 5;
        const sequence = [];
        
        // Create sequence in a single tidy
        for (let i = 0; i < sequenceLength; i++) {
          sequence.push(wrapTensor(tf.randomNormal([inputDim])));
        }

        const wrappedMemory = wrapTensor(tf.zeros([outputDim]));
        
        // Train on sequence
        for (let i = 0; i < sequenceLength - 1; i++) {
          const cost = model.trainStep(sequence[i], sequence[i + 1], wrappedMemory);
          const costShape = unwrapTensor(cost).shape;
          expect(costShape).toEqual([]);
          cost.dispose();
        }

        // Clean up
        sequence.forEach(t => t.dispose());
        wrappedMemory.dispose();
      });
    });
  });

  describe('manifold operations', () => {
    beforeEach(() => {
      model = new TitanMemoryModel({
        inputDim,
        hiddenDim,
        outputDim,
        useManifold: true,
        maxStepSize: 0.1,
        tangentEpsilon: 1e-8
      });
    });

    test('maintains unit norm with standard input', () => {
      let base = tf.randomNormal([inputDim]);
      const baseNorm = base.norm().dataSync()[0];
      base = base.div(tf.scalar(baseNorm + 1e-12));

      let velocity = tf.randomNormal([inputDim]);
      velocity = velocity.mul(tf.scalar(0.05));

      const wrappedBase = wrapTensor(base);
      const wrappedVel = wrapTensor(velocity);

      const result = model.manifoldStep(wrappedBase, wrappedVel);
      const unwrappedResult = unwrapTensor(result);
      const norm = unwrappedResult.norm().dataSync()[0];
      
      expect(Math.abs(norm - 1.0)).toBeLessThan(1e-5);

      // Clean up
      base.dispose();
      velocity.dispose();
      wrappedBase.dispose();
      wrappedVel.dispose();
      result.dispose();
      unwrappedResult.dispose();
    });

    test('handles zero velocity correctly', () => {
      let base = tf.randomNormal([inputDim]);
      const baseNorm = base.norm().dataSync()[0];
      base = base.div(tf.scalar(baseNorm + 1e-12));
      const velocity = tf.zeros([inputDim]);

      const wrappedBase = wrapTensor(base);
      const wrappedVel = wrapTensor(velocity);

      const result = model.manifoldStep(wrappedBase, wrappedVel);
      const unwrappedResult = unwrapTensor(result);
      
      // Should return original base vector
      const diff = tf.sum(tf.sub(unwrappedResult, base)).dataSync()[0];
      expect(Math.abs(diff)).toBeLessThan(1e-5);

      // Clean up
      base.dispose();
      velocity.dispose();
      wrappedBase.dispose();
      wrappedVel.dispose();
      result.dispose();
      unwrappedResult.dispose();
    });

    test('respects maximum step size', () => {
      let base = tf.randomNormal([inputDim]);
      const baseNorm = base.norm().dataSync()[0];
      base = base.div(tf.scalar(baseNorm + 1e-12));

      // Create large velocity
      let velocity = tf.randomNormal([inputDim]);
      velocity = velocity.mul(tf.scalar(1.0)); // Much larger than maxStepSize

      const wrappedBase = wrapTensor(base);
      const wrappedVel = wrapTensor(velocity);

      const result = model.manifoldStep(wrappedBase, wrappedVel);
      const unwrappedResult = unwrapTensor(result);
      
      // Calculate angle between base and result
      const dot = tf.sum(tf.mul(base, unwrappedResult)).dataSync()[0];
      const angle = Math.acos(Math.min(1.0, Math.abs(dot)));
      
      // Angle should not exceed maxStepSize (with small epsilon for floating point precision)
      const epsilon = 1e-6;
      expect(angle).toBeLessThanOrEqual((model.getConfig().maxStepSize || 0.1) + epsilon);

      // Clean up
      base.dispose();
      velocity.dispose();
      wrappedBase.dispose();
      wrappedVel.dispose();
      result.dispose();
      unwrappedResult.dispose();
    });
  });

  describe('model persistence', () => {
    test('saves and loads weights correctly', async () => {
      const model = new TitanMemoryModel({
        inputDim,
        hiddenDim,
        outputDim
      });

      const initialWeights = model.getWeights();
      await model.saveModel('./test-weights.json');

      const loadedModel = new TitanMemoryModel({
        inputDim,
        hiddenDim,
        outputDim
      });
      await loadedModel.loadModel('./test-weights.json');

      const loadedWeights = loadedModel.getWeights();
      expect(loadedWeights).toEqual(initialWeights);
    });

    test('maintains model behavior after load', async () => {
      // Train original model
      const x = wrapTensor(tf.randomNormal([inputDim]));
      const memoryState = wrapTensor(tf.zeros([outputDim]));
      const { predicted: originalPrediction } = model.forward(x, memoryState);

      await model.saveModel('./test-weights.json');

      // Load into new model
      const loadedModel = new TitanMemoryModel({
        inputDim,
        hiddenDim,
        outputDim
      });
      await loadedModel.loadModel('./test-weights.json');

      // Compare predictions
      const { predicted: loadedPrediction } = loadedModel.forward(x, memoryState);
      
      const originalData = unwrapTensor(originalPrediction).dataSync();
      const loadedData = unwrapTensor(loadedPrediction).dataSync();
      
      for (let i = 0; i < originalData.length; i++) {
        expect(originalData[i]).toBeCloseTo(loadedData[i], 5);
      }

      // Clean up
      x.dispose();
      memoryState.dispose();
      originalPrediction.dispose();
      loadedPrediction.dispose();
    });

    test('handles invalid file paths', async () => {
      await expect(model.loadModel('./nonexistent.json'))
        .rejects.toThrow();
    });
  });
});
