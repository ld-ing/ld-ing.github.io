import { GPGPUProgram } from './gpgpu_math';
export declare class Pad1DProgram implements GPGPUProgram {
    variableNames: string[];
    outputShape: number[];
    userCode: string;
    rank: number;
    constructor(xShape: number[], paddings: [number, number], constantValue: number);
}
export declare class Pad2DProgram implements GPGPUProgram {
    variableNames: string[];
    outputShape: number[];
    userCode: string;
    rank: number;
    constructor(xShape: number[], paddings: [[number, number], [number, number]], constantValue: number);
}
