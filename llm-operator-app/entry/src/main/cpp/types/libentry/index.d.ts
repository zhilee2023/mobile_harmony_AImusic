export const inspect: (root: string, model: number) => string;
export const start: (root: string, model: number, options: string) => Promise<string>;
export const poll: () => string;
export const cancel: () => string;
