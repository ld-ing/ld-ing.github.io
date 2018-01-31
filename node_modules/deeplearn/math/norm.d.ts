import { NDArray } from './ndarray';
export declare class Ops {
    static norm(x: NDArray, ord?: number | 'euclidean' | 'fro', axis?: number | number[], keepDims?: boolean): NDArray;
}
