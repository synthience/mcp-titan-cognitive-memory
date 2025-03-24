import express from 'express';
import bodyParser from 'body-parser';
import * as tf from '@tensorflow/tfjs';
import { TitanMemoryModel } from './model.js';
import { wrapTensor } from './types.js';
import { Server } from 'http';
import WebSocket from 'ws';

export class TitanExpressServer {
  private app: express.Application;
  private server: Server | null = null;
  private model: TitanMemoryModel | null = null;
  private memoryVec: tf.Variable | null = null;
  private port: number;
  private wss: WebSocket.Server | null = null;

  constructor(port: number = 3000) {
    this.port = port;
    this.app = express();
    this.setupMiddleware();
    this.setupRoutes();
    this.setupWebSocket();
  }

  private setupMiddleware() {
    this.app.use(bodyParser.json());
  }

  private setupRoutes() {
    // Initialize model
    this.app.post('/init', (req, res) => {
      try {
        const config = req.body || {};
        this.model = new TitanMemoryModel(config);
        
        // Initialize memory vector
        if (this.memoryVec) {
          this.memoryVec.dispose();
        }
        const memDim = this.model.getConfig().outputDim || 64; // Interpreted as memory dimension
        this.memoryVec = tf.variable(tf.zeros([memDim]));

        return res.json({ 
          message: 'Model initialized', 
          config: this.model.getConfig() 
        });
      } catch (error) {
        return res.status(500).json({ 
          error: 'Failed to initialize model',
          details: error instanceof Error ? error.message : String(error)
        });
      }
    });

    // Train step
    this.app.post('/trainStep', async (req, res) => {
      if (!this.model || !this.memoryVec) {
        return res.status(400).json({ error: 'Model not initialized' });
      }

      try {
        const { x_t, x_next } = req.body;
        if (!x_t || !x_next) {
          return res.status(400).json({ error: 'Missing x_t / x_next' });
        }

        // Convert to tensors and wrap them
        const x_tT = wrapTensor(tf.tensor1d(x_t));
        const x_nextT = wrapTensor(tf.tensor1d(x_next));
        const memoryT = wrapTensor(this.memoryVec);

        // Run training step
        const cost = this.model.trainStep(x_tT, x_nextT, memoryT);

        // Forward pass results
        const { predicted, newMemory, surprise } = this.model.forward(x_tT, memoryT);

        // Extract values
        const costVal = cost.dataSync()[0];
        const predVal = predicted.dataSync();
        const surVal = surprise.dataSync()[0];

        // Update memory
        this.memoryVec.assign(tf.tensor(newMemory.dataSync()));

        // Cleanup
        x_tT.dispose();
        x_nextT.dispose();
        memoryT.dispose();
        predicted.dispose();
        newMemory.dispose();
        surprise.dispose();
        cost.dispose();

        return res.json({
          cost: costVal,
          predicted: Array.from(predVal),
          surprise: surVal
        });
      } catch (error) {
        return res.status(500).json({ 
          error: 'Training step failed',
          details: error instanceof Error ? error.message : String(error)
        });
      }
    });

    // Forward pass
    this.app.post('/forward', async (req, res) => {
      if (!this.model || !this.memoryVec) {
        return res.status(400).json({ error: 'Model not initialized' });
      }

      try {
        const { x } = req.body;
        if (!x) {
          return res.status(400).json({ error: 'Missing input vector' });
        }

        // Convert to tensors
        const xT = wrapTensor(tf.tensor1d(x));
        const memoryT = wrapTensor(this.memoryVec);

        // Run forward pass
        const { predicted, newMemory, surprise } = this.model.forward(xT, memoryT);

        // Extract values
        const predVal = predicted.dataSync();
        const memVal = newMemory.dataSync();
        const surVal = surprise.dataSync()[0];

        // Update memory
        this.memoryVec.assign(tf.tensor(newMemory.dataSync()));

        // Cleanup
        xT.dispose();
        memoryT.dispose();
        predicted.dispose();
        newMemory.dispose();
        surprise.dispose();

        return res.json({
          predicted: Array.from(predVal),
          memory: Array.from(memVal),
          surprise: surVal
        });
      } catch (error) {
        return res.status(500).json({ 
          error: 'Forward pass failed',
          details: error instanceof Error ? error.message : String(error)
        });
      }
    });

    // Save model
    this.app.post('/save', async (req, res) => {
      if (!this.model) {
        return res.status(400).json({ error: 'Model not initialized' });
      }

      try {
        const { path } = req.body;
        if (!path) {
          return res.status(400).json({ error: 'Missing path' });
        }

        await this.model.saveModel(path);
        return res.json({ message: 'Model saved' });
      } catch (error) {
        return res.status(500).json({ 
          error: 'Failed to save model',
          details: error instanceof Error ? error.message : String(error)
        });
      }
    });

    // Load model
    this.app.post('/load', async (req, res) => {
      if (!this.model) {
        return res.status(400).json({ error: 'Model not initialized' });
      }

      try {
        const { path } = req.body;
        if (!path) {
          return res.status(400).json({ error: 'Missing path' });
        }

        await this.model.loadModel(path);
        return res.json({ message: 'Model loaded' });
      } catch (error) {
        return res.status(500).json({ 
          error: 'Failed to load model',
          details: error instanceof Error ? error.message : String(error)
        });
      }
    });

    // Status
    this.app.get('/status', (req, res) => {
      if (!this.model) {
        return res.status(200).json({ status: 'No model' });
      }
      return res.json(this.model.getConfig());
    });
  }

  private setupWebSocket() {
    this.wss = new WebSocket.Server({ port: this.port + 1 });

    this.wss.on('connection', (ws) => {
      ws.on('message', (message) => {
        const data = JSON.parse(message.toString());
        const { action, payload } = data;

        switch (action) {
          case 'forward':
            if (!this.model || !this.memoryVec) {
              ws.send(JSON.stringify({ error: 'Model not initialized' }));
              return;
            }

            const xT = wrapTensor(tf.tensor1d(payload.x));
            const memoryT = wrapTensor(this.memoryVec);

            const { predicted, newMemory, surprise } = this.model.forward(xT, memoryT);

            const predVal = predicted.dataSync();
            const memVal = newMemory.dataSync();
            const surVal = surprise.dataSync()[0];

            this.memoryVec.assign(tf.tensor(newMemory.dataSync()));

            xT.dispose();
            memoryT.dispose();
            predicted.dispose();
            newMemory.dispose();
            surprise.dispose();

            ws.send(JSON.stringify({
              predicted: Array.from(predVal),
              memory: Array.from(memVal),
              surprise: surVal
            }));
            break;

          default:
            ws.send(JSON.stringify({ error: 'Unknown action' }));
        }
      });
    });
  }

  public start(): Promise<void> {
    return new Promise((resolve) => {
      this.server = this.app.listen(this.port, () => {
        console.log(`[TitanServer] Listening on port ${this.port}`);
        resolve();
      });
    });
  }

  public async stop(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.server) {
        this.server.close((err) => {
          if (err) {
            reject(err);
          } else {
            if (this.memoryVec) {
              this.memoryVec.dispose();
              this.memoryVec = null;
            }
            this.server = null;
            resolve();
          }
        });
      } else {
        resolve();
      }
    });
  }
}
