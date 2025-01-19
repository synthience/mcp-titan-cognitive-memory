#!/usr/bin/env node
import '@tensorflow/tfjs-node'; // Import and register the Node.js backend
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { CallToolRequestSchema, CallToolResultSchema, ErrorCode } from '@modelcontextprotocol/sdk/types.js';
import * as tf from '@tensorflow/tfjs';
import { TitanMemoryModel } from './model.js';
import { wrapTensor } from './types.js';
class TitanMemoryServer {
    constructor() {
        // Initialize MCP server metadata only
        this.server = new Server({
            name: 'titan-memory-server',
            version: '0.1.0',
        }, {
            capabilities: {
                tools: {
                    train: {
                        name: 'train',
                        description: 'Train the model on a sequence of vectors (each 64D by default).',
                        parameters: {
                            type: 'object',
                            properties: {
                                sequence: {
                                    type: 'array',
                                    items: {
                                        type: 'array',
                                        items: { type: 'number' }
                                    }
                                }
                            },
                            required: ['sequence']
                        }
                    }
                },
            },
        });
        // Initialize model with default config
        this.model = new TitanMemoryModel({ inputDim: 64, outputDim: 64 });
        this.memoryVec = tf.variable(tf.zeros([this.model.getConfig().outputDim || 64]));
        this.setupToolHandlers();
        // Error handling
        this.server.onerror = (error) => console.error('[MCP Error]', error);
        process.on('SIGINT', async () => {
            await this.cleanup();
            process.exit(0);
        });
    }
    setupToolHandlers() {
        this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
            if (request.params.name !== 'train') {
                return CallToolResultSchema.parse({
                    error: {
                        code: ErrorCode.MethodNotFound,
                        message: `Unknown tool: ${request.params.name}`
                    }
                });
            }
            if (!request.params.arguments || !Array.isArray(request.params.arguments.sequence)) {
                return CallToolResultSchema.parse({
                    result: {
                        content: [
                            {
                                type: 'error',
                                text: 'Invalid sequence format: must be an array of 64-dimensional vectors'
                            }
                        ]
                    }
                });
            }
            try {
                const sequence = request.params.arguments.sequence;
                const outputs = [];
                const metrics = [];
                for (let t = 0; t < sequence.length; t++) {
                    const x_t = tf.tidy(() => {
                        const buffer = tf.buffer([1, 64]);
                        for (let i = 0; i < 64; i++) {
                            buffer.set(sequence[t][i], 0, i);
                        }
                        return buffer.toTensor().squeeze();
                    });
                    // Next element for training if available
                    const x_next = t < sequence.length - 1
                        ? tf.tensor1d(sequence[t + 1])
                        : x_t;
                    // Wrap
                    const wrappedX = wrapTensor(x_t);
                    const wrappedNext = wrapTensor(x_next);
                    const wrappedMemory = wrapTensor(this.memoryVec);
                    // Training step
                    const cost = this.model.trainStep(wrappedX, wrappedNext, wrappedMemory);
                    // Forward pass results (to track metrics)
                    const { predicted, newMemory, surprise } = this.model.forward(wrappedX, wrappedMemory);
                    // Update memory
                    this.memoryVec.assign(tf.tensor(newMemory.dataSync()));
                    // Store results
                    outputs.push(tf.tensor(predicted.dataSync()));
                    metrics.push({
                        step: t,
                        cost: cost.dataSync()[0],
                        surprise: surprise.dataSync()[0]
                    });
                    // Cleanup
                    wrappedX.dispose();
                    wrappedNext.dispose();
                    wrappedMemory.dispose();
                    x_t.dispose();
                    if (t < sequence.length - 1)
                        x_next.dispose();
                    predicted.dispose();
                    newMemory.dispose();
                    surprise.dispose();
                    cost.dispose();
                }
                // Stack outputs for final
                const finalOutput = tf.stack(outputs); // shape: [sequenceLen, inputDim=64]
                const result = {
                    shape: finalOutput.shape,
                    output: Array.from(finalOutput.dataSync()),
                    metrics
                };
                finalOutput.dispose();
                return CallToolResultSchema.parse({
                    content: [{
                            type: 'text',
                            text: JSON.stringify(result, null, 2)
                        }]
                });
            }
            catch (error) {
                const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
                return CallToolResultSchema.parse({
                    result: {
                        content: [
                            {
                                type: 'error',
                                text: `Processing error: ${errorMessage}`
                            }
                        ]
                    }
                });
            }
        });
    }
    async cleanup() {
        if (this.memoryVec) {
            this.memoryVec.dispose();
        }
    }
    async run() {
        await this.server.connect(new StdioServerTransport());
        console.log('Titan Memory MCP server running on stdio');
    }
}
const server = new TitanMemoryServer();
server.run().catch(console.error);
//# sourceMappingURL=index.js.map