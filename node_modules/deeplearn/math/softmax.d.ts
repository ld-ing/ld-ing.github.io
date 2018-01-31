import { NDArray } from './ndarray';
export declare class Ops {
    static softmax<T extends NDArray>(logits: T, dim?: number): T;
    static softmaxCrossEntropy<T extends NDArray, O extends NDArray>(labels: T, logits: T, dim?: number): O;
}
