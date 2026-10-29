export const memoryProbe: () => string;
export const runVaeStream: (directory: string, precision: string) => Promise<string>;
export const runThreeModuleChain: (directory: string, policy: string) => Promise<string>;
