import request from 'supertest';
import express from 'express';
import { TitanExpressServer } from '../server.js';

describe('TitanExpressServer Tests', () => {
  let server: TitanExpressServer;
  let app: express.Application;

  beforeAll(() => {
    server = new TitanExpressServer(3001); // Use a test port
    // Access the internal Express app for test calls:
    app = (server as any).app;
  });

  afterAll(() => {
    server.stop();
  });

  test('Initialize model with config', async () => {
    const config = {
      inputDim: 32,
      hiddenDim: 16,
      outputDim: 32,
      learningRate: 0.001
    };

    const response = await request(app)
      .post('/init')
      .send(config);

    expect(response.status).toBe(200);
    expect(response.body.config).toMatchObject(config);
  });

  test('Training step with valid input', async () => {
    // Re-init model
    await request(app)
      .post('/init')
      .send({
        inputDim: 64,
        outputDim: 64
      });

    const x_t = Array(64).fill(0).map(() => Math.random());
    const x_next = Array(64).fill(0).map(() => Math.random());

    const response = await request(app)
      .post('/trainStep')
      .send({ x_t, x_next });

    expect(response.status).toBe(200);
    expect(response.body).toHaveProperty('cost');
    expect(response.body).toHaveProperty('predicted');
    expect(response.body).toHaveProperty('surprise');
    expect(response.body.predicted).toHaveLength(64);
  });

  test('Forward pass with valid input', async () => {
    // Re-init model
    await request(app)
      .post('/init')
      .send({
        inputDim: 64,
        outputDim: 64
      });

    const x_t = Array(64).fill(0).map(() => Math.random());

    const response = await request(app)
      .post('/forward')
      .send({ x: x_t });

    expect(response.status).toBe(200);
    expect(response.body).toHaveProperty('predicted');
    expect(response.body).toHaveProperty('surprise');
    expect(response.body.predicted).toHaveLength(64);
  });

  test('Save and load model weights', async () => {
    // First initialize model
    await request(app)
      .post('/init')
      .send({ inputDim: 64, outputDim: 64 });

    // Save weights
    const saveResponse = await request(app)
      .post('/save')
      .send({ path: 'file://./test-weights' });

    expect(saveResponse.status).toBe(200);
    expect(saveResponse.body.message).toContain('Model saved');

    // Load weights
    const loadResponse = await request(app)
      .post('/load')
      .send({ path: 'file://./test-weights' });

    expect(loadResponse.status).toBe(200);
    expect(loadResponse.body.message).toContain('Model loaded');
  });

  test('Get model status', async () => {
    // Re-init model with specific config
    const config = {
      inputDim: 32,
      hiddenDim: 16,
      outputDim: 32,
      learningRate: 0.001
    };

    await request(app)
      .post('/init')
      .send(config);

    const response = await request(app)
      .get('/status');

    expect(response.status).toBe(200);
    expect(response.body).toMatchObject(config);
  });

  test('Handle errors gracefully', async () => {
    // Train step called with invalid vector dimensions
    const response = await request(app)
      .post('/trainStep')
      .send({
        x_t: [1, 2], // Too few
        x_next: [3, 4]
      });

    expect(response.status).toBe(500);
    expect(response.body).toHaveProperty('error');
  });

  test('Handles unknown tool in switch statement', async () => {
    const response = await request(app)
      .post('/unknown_tool')
      .send({
        params: {
          name: 'unknown_tool',
          arguments: {}
        }
      });

    expect(response.status).toBe(404);
    expect(response.body).toHaveProperty('error');
    expect(response.body.error.code).toBe('MethodNotFound');
    expect(response.body.error.message).toBe('Unknown tool: unknown_tool');
  });

  test('CallToolResultSchema.parse return statement', async () => {
    const response = await request(app)
      .post('/init')
      .send({
        params: {
          name: 'init_model',
          arguments: {}
        }
      });

    expect(response.status).toBe(200);
    expect(response.body).toHaveProperty('content');
    expect(response.body.content[0].type).toBe('text');
    expect(response.body.content[0].text).toContain('Model initialized');
  });

  test('Store memory state in LLM cache', async () => {
    // Re-init model
    await request(app)
      .post('/init')
      .send({
        inputDim: 64,
        outputDim: 64
      });

    const response = await request(app)
      .post('/store_memory_state')
      .send({ key: 'test_key' });

    expect(response.status).toBe(200);
    expect(response.body.message).toContain('Memory state stored');
  });

  test('Retrieve memory state from LLM cache', async () => {
    // Re-init model
    await request(app)
      .post('/init')
      .send({
        inputDim: 64,
        outputDim: 64
      });

    const response = await request(app)
      .post('/retrieve_memory_state')
      .send({ key: 'test_key' });

    expect(response.status).toBe(200);
    expect(response.body.message).toContain('Memory state retrieved');
  });

  test('WebSocket forward pass', (done) => {
    const ws = new WebSocket('ws://localhost:3002');

    ws.on('open', () => {
      ws.send(JSON.stringify({
        action: 'forward',
        payload: { x: Array(64).fill(0).map(() => Math.random()) }
      }));
    });

    ws.on('message', (message) => {
      const data = JSON.parse(message.toString());
      expect(data).toHaveProperty('predicted');
      expect(data).toHaveProperty('memory');
      expect(data).toHaveProperty('surprise');
      expect(data.predicted).toHaveLength(64);
      ws.close();
      done();
    });

    ws.on('error', (error) => {
      done(error);
    });
  });
});
