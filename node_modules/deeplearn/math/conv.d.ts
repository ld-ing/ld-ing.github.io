import { Array1D, Array2D, Array3D, Array4D } from './ndarray';
export declare class Ops {
    static conv1d<T extends Array2D | Array3D>(input: T, filter: Array3D, bias: Array1D | null, stride: number, pad: 'valid' | 'same' | number, dimRoundingMode?: 'floor' | 'round' | 'ceil'): T;
    static conv2d<T extends Array3D | Array4D>(x: T, filter: Array4D, bias: Array1D | null, strides: [number, number] | number, pad: 'valid' | 'same' | number, dimRoundingMode?: 'floor' | 'round' | 'ceil'): T;
    static conv2dDerInput<T extends Array3D | Array4D>(xShape: [number, number, number, number] | [number, number, number], dy: T, filter: Array4D, strides: [number, number] | number, pad: 'valid' | 'same' | number, dimRoundingMode?: 'floor' | 'round' | 'ceil'): T;
    static conv2dDerBias(dy: Array3D | Array4D): Array1D;
    static conv2dDerFilter<T extends Array3D | Array4D>(x: T, dy: T, filterShape: [number, number, number, number], strides: [number, number] | number, pad: 'valid' | 'same' | number, dimRoundingMode?: 'floor' | 'round' | 'ceil'): Array4D;
    static conv2dTranspose<T extends Array3D | Array4D>(x: T, filter: Array4D, outputShape: [number, number, number, number] | [number, number, number], strides: [number, number] | number, pad: 'valid' | 'same' | number, dimRoundingMode?: 'floor' | 'round' | 'ceil'): T;
    static depthwiseConv2D<T extends Array3D | Array4D>(input: T, filter: Array4D, strides: [number, number] | number, pad: 'valid' | 'same' | number, rates?: [number, number] | number, dimRoundingMode?: 'floor' | 'round' | 'ceil'): T;
}
