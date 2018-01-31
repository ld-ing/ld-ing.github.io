import { Array3D, Array4D } from './ndarray';
export declare class Ops {
    static maxPool<T extends Array3D | Array4D>(x: T, filterSize: [number, number] | number, strides: [number, number] | number, pad: 'valid' | 'same' | number, dimRoundingMode?: 'floor' | 'round' | 'ceil'): T;
    static maxPoolBackprop<T extends Array3D | Array4D>(dy: T, input: T, filterSize: [number, number] | number, strides: [number, number] | number, pad: 'valid' | 'same' | number, dimRoundingMode?: 'floor' | 'round' | 'ceil'): T;
    static minPool<T extends Array3D | Array4D>(input: T, filterSize: [number, number] | number, strides: [number, number] | number, pad: 'valid' | 'same' | number, dimRoundingMode?: 'floor' | 'round' | 'ceil'): T;
    static avgPool<T extends Array3D | Array4D>(x: T, filterSize: [number, number] | number, strides: [number, number] | number, pad: 'valid' | 'same' | number, dimRoundingMode?: 'floor' | 'round' | 'ceil'): T;
    static avgPoolBackprop<T extends Array3D | Array4D>(dy: T, input: T, filterSize: [number, number] | number, strides: [number, number] | number, pad: 'valid' | 'same' | number): T;
}
