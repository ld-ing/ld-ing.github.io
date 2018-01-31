import { NDArray } from './ndarray';
export declare class Ops {
    static notEqual<T extends NDArray>(a: NDArray, b: NDArray): T;
    static notEqualStrict<T extends NDArray>(a: T, b: T): T;
    static less<T extends NDArray>(a: NDArray, b: NDArray): T;
    static lessStrict<T extends NDArray>(a: T, b: T): T;
    static equal<T extends NDArray>(a: NDArray, b: NDArray): T;
    static equalStrict<T extends NDArray>(a: T, b: T): T;
    static lessEqual<T extends NDArray>(a: NDArray, b: NDArray): T;
    static lessEqualStrict<T extends NDArray>(a: T, b: T): T;
    static greater<T extends NDArray>(a: NDArray, b: NDArray): T;
    static greaterStrict<T extends NDArray>(a: T, b: T): T;
    static greaterEqual<T extends NDArray>(a: NDArray, b: NDArray): T;
    static greaterEqualStrict<T extends NDArray>(a: T, b: T): T;
}
