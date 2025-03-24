#!/usr/bin/env node
import '@tensorflow/tfjs-node';  // Import and register the Node.js backend
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  CallToolResultSchema,
  ErrorCode
} from '@modelcontextprotocol/sdk/types.js';
import * as tf from '@tensorflow/tfjs';
import { TitanMemoryModel } from './model.js';
import { wrapTensor, unwrapTensor } from './types.js';

class TitanMemoryServer {
  private server: Server;
  private model: TitanMemoryModel | null = null;
  private memoryVec: tf.Variable | null = null;

  constructor() {
    // Initialize MCP server metadata
    this.server = new Server(
      {
        name: 'titan-memory-server',
        version: '0.1.0',
      },
      {
        capabilities: {
          tools: {
            init_model: {
              name: 'init_model',
              description: 'Initialize the Titan Memory model with optional configuration.',
              parameters: {
                type: 'object',
                properties: {
                  inputDim: {
                    type: 'number',
                    description: 'Input dimension (default: 64)'
                  },
                  outputDim: {
                    type: 'number',
                    description: 'Output/Memory dimension (default: 64)'
                  }
                }
              }
            },
            train_step: {
              name: 'train_step',
              description: 'Perform a single training step with current and next state vectors.',
              parameters: {
                type: 'object',
                properties: {
                  x_t: {
                    type: 'array',
                    items: { type: 'number' },
                    description: 'Current state vector'
                  },
                  x_next: {
                    type: 'array',
                    items: { type: 'number' },
                    description: 'Next state vector'
                  }
                },
                required: ['x_t', 'x_next']
              }
            },
            forward_pass: {
              name: 'forward_pass',
              description: 'Run a forward pass through the model with an input vector.',
              parameters: {
                type: 'object',
                properties: {
                  x: {
                    type: 'array',
                    items: { type: 'number' },
                    description: 'Input vector'
                  }
                },
                required: ['x']
              }
            },
            save_model: {
              name: 'save_model',
              description: 'Save the model to a specified path.',
              parameters: {
                type: 'object',
                properties: {
                  path: {
                    type: 'string',
                    description: 'Path to save the model'
                  }
                },
                required: ['path']
              }
            },
            load_model: {
              name: 'load_model',
              description: 'Load the model from a specified path.',
              parameters: {
                type: 'object',
                properties: {
                  path: {
                    type: 'string',
                    description: 'Path to load the model from'
                  }
                },
                required: ['path']
              }
            },
            get_status: {
              name: 'get_status',
              description: 'Get current model status and configuration.',
              parameters: {
                type: 'object',
                properties: {}
              }
            },
            train_sequence: {
              name: 'train_sequence',
              description: 'Train the model on a sequence of vectors.',
              parameters: {
                type: 'object',
                properties: {
                  sequence: {
                    type: 'array',
                    items: {
                      type: 'array',
                      items: { type: 'number' }
                    },
                    description: 'Sequence of vectors to train on'
                  }
                },
                required: ['sequence']
              }
            },
            store_memory_state: {
              name: 'store_memory_state',
              description: 'Store the current memory state in the LLM cache.',
              parameters: {
                type: 'object',
                properties: {
                  key: {
                    type: 'string',
                    description: 'Key to store the memory state under'
                  }
                },
                required: ['key']
              }
            },
            retrieve_memory_state: {
              name: 'retrieve_memory_state',
              description: 'Retrieve a memory state from the LLM cache.',
              parameters: {
                type: 'object',
                properties: {
                  key: {
                    type: 'string',
                    description: 'Key to retrieve the memory state from'
                  }
                },
                required: ['key']
              }
            }
          },
        },
      }
    );

    this.setupToolHandlers();
    
    // Error handling
    this.server.onerror = (error) => console.error('[MCP Error]', error);
    process.on('SIGINT', async () => {
      await this.cleanup();
      process.exit(0);
    });
  }

  private setupToolHandlers() {
    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      try {
        switch (request.params.name) {
          case 'init_model': {
            const config = request.params.arguments || {};
            this.model = new TitanMemoryModel(config);

            if (this.memoryVec) {
              this.memoryVec.dispose();
            }
            const memDim = this.model.getConfig().outputDim || 64;
            this.memoryVec = tf.variable(tf.zeros([memDim]));

            return CallToolResultSchema.parse({
              content: [{
                type: 'text',
                text: JSON.stringify({
                  message: 'Model initialized',
                  config: this.model.getConfig()
                }, null, 2)
              }]
            });
          }

          case 'train_step': {
            if (!this.model || !this.memoryVec) {
              throw new Error('Model not initialized');
            }

            const args = request.params.arguments as { x_t?: number[], x_next?: number[] };
            if (!args.x_t || !args.x_next) {
              throw new Error('Missing x_t / x_next');
            }
            const { x_t, x_next } = args;

            const x_tT = wrapTensor(tf.tensor1d(x_t));
            const x_nextT = wrapTensor(tf.tensor1d(x_next));
            const memoryT = wrapTensor(this.memoryVec);

            const cost = this.model.trainStep(x_tT, x_nextT, memoryT);
            const { predicted, newMemory, surprise } = this.model.forward(x_tT, memoryT);

            const result = {
              cost: cost.dataSync()[0],
              predicted: Array.from(predicted.dataSync()),
              surprise: surprise.dataSync()[0]
            };

            this.memoryVec.assign(tf.tensor(newMemory.dataSync()));

            // Cleanup
            [x_tT, x_nextT, memoryT, predicted, newMemory, surprise, cost].forEach(t => t.dispose());

            return CallToolResultSchema.parse({
              content: [{
                type: 'text',
                text: JSON.stringify(result, null, 2)
              }]
            });
          }

          case 'forward_pass': {
            if (!this.model || !this.memoryVec) {
              throw new Error('Model not initialized');
            }
            const args = request.params.arguments as { x?: number[] };
            if (!args.x) {
              throw new Error('Missing input vector');
            }

            const xT = wrapTensor(tf.tensor1d(args.x));
            const memoryT = wrapTensor(this.memoryVec);

            const { predicted, newMemory, surprise } = this.model.forward(xT, memoryT);

            const result = {
              predicted: Array.from(predicted.dataSync()),
              memory: Array.from(newMemory.dataSync()),
              surprise: surprise.dataSync()[0]
            };

            this.memoryVec.assign(tf.tensor(newMemory.dataSync()));

            // Cleanup
            [xT, memoryT, predicted, newMemory, surprise].forEach(t => t.dispose());

            return CallToolResultSchema.parse({
              content: [{
                type: 'text',
                text: JSON.stringify(result, null, 2)
              }]
            });
          }

          case 'save_model': {
            if (!this.model) {
              throw new Error('Model not initialized');
            }
            const args = request.params.arguments as { path?: string };
            if (!args.path) {
              throw new Error('Missing path');
            }

            await this.model.saveModel(args.path);

            return CallToolResultSchema.parse({
              content: [{
                type: 'text',
                text: JSON.stringify({ message: 'Model saved' }, null, 2)
              }]
            });
          }

          case 'load_model': {
            if (!this.model) {
              throw new Error('Model not initialized');
            }
            const args = request.params.arguments as { path?: string };
            if (!args.path) {
              throw new Error('Missing path');
            }

            await this.model.loadModel(args.path);

            return CallToolResultSchema.parse({
              content: [{
                type: 'text',
                text: JSON.stringify({ message: 'Model loaded' }, null, 2)
              }]
            });
          }

          case 'get_status': {
            const status = this.model
              ? this.model.getConfig()
              : { status: 'No model initialized' };

            return CallToolResultSchema.parse({
              content: [{
                type: 'text',
                text: JSON.stringify(status, null, 2)
              }]
            });
          }

          case 'train_sequence': {
            if (!this.model || !this.memoryVec) {
              throw new Error('Model not initialized');
            }
            const args = request.params.arguments as { sequence?: number[][] };
            if (!args.sequence || !Array.isArray(args.sequence)) {
              throw new Error('Invalid sequence format');
            }
            const { sequence } = args;

            const outputs: tf.Tensor[] = [];
            const metrics: any[] = [];

            for (let t = 0; t < sequence.length; t++) {
              const x_t = tf.tidy(() => {
                const buffer = tf.buffer([1, this.model!.getConfig().inputDim || 64]);
                for (let i = 0; i < sequence[t].length; i++) {
                  buffer.set(sequence[t][i], 0, i);
                }
                return buffer.toTensor().squeeze();
              });

              const x_next = t < sequence.length - 1
                ? tf.tensor1d(sequence[t + 1])
                : x_t;

              const wrappedX = wrapTensor(x_t);
              const wrappedNext = wrapTensor(x_next);
              const wrappedMemory = wrapTensor(this.memoryVec!);

              const cost = this.model.trainStep(wrappedX, wrappedNext, wrappedMemory);
              const { predicted, newMemory, surprise } = this.model.forward(wrappedX, wrappedMemory);

              this.memoryVec!.assign(tf.tensor(newMemory.dataSync()));

              outputs.push(tf.tensor(predicted.dataSync()));
              metrics.push({
                step: t,
                cost: cost.dataSync()[0],
                surprise: surprise.dataSync()[0]
              });

              // Cleanup
              [wrappedX, wrappedNext, wrappedMemory, x_t, predicted, newMemory, surprise, cost].forEach(t => t.dispose());
              if (t < sequence.length - 1) x_next.dispose();
            }

            const finalOutput = tf.stack(outputs);
            const result = {
              shape: finalOutput.shape,
              output: Array.from(finalOutput.dataSync()),
              metrics
            };

            finalOutput.dispose();
            outputs.forEach(t => t.dispose());

            return CallToolResultSchema.parse({
              content: [{
                type: 'text',
                text: JSON.stringify(result, null, 2)
              }]
            });
          }

          case 'store_memory_state': {
            if (!this.model || !this.memoryVec) {
              throw new Error('Model not initialized');
            }
            const args = request.params.arguments as { key?: string };
            if (!args.key) {
              throw new Error('Missing key');
            }

            // Store the memory state in the LLM cache
            const memoryState = this.memoryVec.dataSync();
            // Implement the logic to store the memory state in the LLM cache using the provided key

            return CallToolResultSchema.parse({
              content: [{
                type: 'text',
                text: JSON.stringify({ message: 'Memory state stored' }, null, 2)
              }]
            });
          }

          case 'retrieve_memory_state': {
            if (!this.model || !this.memoryVec) {
              throw new Error('Model not initialized');
            }
            const args = request.params.arguments as { key?: string };
            if (!args.key) {
              throw new Error('Missing key');
            }

            // Retrieve the memory state from the LLM cache
            // Implement the logic to retrieve the memory state from the LLM cache using the provided key
            const memoryState = []; // Replace with the actual retrieved memory state

            this.memoryVec.assign(tf.tensor(memoryState));

            return CallToolResultSchema.parse({
              content: [{
                type: 'text',
                text: JSON.stringify({ message: 'Memory state retrieved' }, null, 2)
              }]
            });
          }

          default:
            return CallToolResultSchema.parse({
              error: {
                code: ErrorCode.MethodNotFound,
                message: `Unknown tool: ${request.params.name}`
              }
            });
        }
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
        return CallToolResultSchema.parse({
          error: {
            code: ErrorCode.InternalError,
            message: `Error: ${errorMessage}`
          }
        });
      }
    });
  }

  private async cleanup() {
    if (this.memoryVec) {
      this.memoryVec.dispose();
      this.memoryVec = null;
    }
  }

  public async run() {
    await this.server.connect(new StdioServerTransport());
    console.log('Titan Memory MCP server running on stdio');
  }
}

const server = new TitanMemoryServer();
server.run().catch(console.error);
