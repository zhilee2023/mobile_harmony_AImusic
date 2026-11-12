export const start: (root: string, model: number, frames: number, transport: boolean, lyrics: string, npy: string, options: string) => Promise<string>;
export const styles: (root: string) => string;
export const restore: (root: string) => string;
export const poll: () => string;
export const cancel: () => string;
export const inspect: (root: string, model: number) => string;
