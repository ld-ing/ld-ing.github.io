import { NDArray, Scalar } from './ndarray';
export declare class Ops {
    static logSumExp<T extends NDArray>(input: NDArray, axis?: number | number[], keepDims?: boolean): T;
    static sum<T extends NDArray>(x: NDArray, axis?: number | number[], keepDims?: boolean): T;
    static mean<T extends NDArray>(x: NDArray, axis?: number | number[], keepDims?: boolean): T;
    static min<T extends NDArray>(x: NDArray, axis?: number | number[], keepDims?: boolean): T;
    static max<T extends NDArray>(x: NDArray, axis?: number | number[], keepDims?: boolean): T;
    static argMin<T extends NDArray>(x: NDArray, axis?: number): T;
    static argMax<T extends NDArray>(x: NDArray, axis?: number): T;
    static argMaxEquals(x1: NDArray, x2: NDArray): Scalar;
    static moments(x: NDArray, axis?: number | number[], keepDims?: boolean): {
        mean: NDArray;
        variance: NDArray;
    };
}
