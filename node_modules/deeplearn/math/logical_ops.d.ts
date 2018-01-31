import { NDArray } from './ndarray';
export declare class Ops {
    static logicalAnd<T extends NDArray>(a: NDArray, b: NDArray): T;
    static logicalOr<T extends NDArray>(a: NDArray, b: NDArray): T;
    static where<T extends NDArray>(condition: NDArray, a: T, b: T): T;
}
