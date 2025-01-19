export declare class TitanExpressServer {
    private app;
    private server;
    private model;
    private memoryVec;
    private port;
    constructor(port?: number);
    private setupMiddleware;
    private setupRoutes;
    start(): Promise<void>;
    stop(): Promise<void>;
}
