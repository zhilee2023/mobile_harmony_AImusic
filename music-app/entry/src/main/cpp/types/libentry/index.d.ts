export const generate: (directory: string, lyrics: string, style: string, frames: number) => Promise<string>;
export const importAsset: (directory: string, name: string, fd: number) => Promise<string>;
export const inspect: (directory: string) => string;
export const progress: (directory: string) => string;
export const cancel: () => string;
