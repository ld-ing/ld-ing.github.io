import { NDArray, Scalar } from './ndarray';
export declare class Ops {
    static add<T extends NDArray>(a: NDArray, b: NDArray): T;
    static addStrict<T extends NDArray>(a: T, b: T): T;
    static sub<T extends NDArray>(a: NDArray, b: NDArray): T;
    static subStrict<T extends NDArray>(a: T, b: T): T;
    static pow<T extends NDArray>(base: NDArray, exp: NDArray): T;
    static powStrict<T extends NDArray>(base: T, exp: NDArray): T;
    static mul<T extends NDArray>(a: NDArray, b: NDArray): T;
    static elementWiseMul<T extends NDArray>(a: T, b: T): T;
    static mulStrict<T extends NDArray>(a: T, b: T): T;
    static div<T extends NDArray>(a: NDArray, b: NDArray): T;
    static divStrict<T extends NDArray>(a: T, b: T): T;
    static scalarDividedByArray<T extends NDArray>(c: Scalar, a: T): T;
    static arrayDividedByScalar<T extends NDArray>(a: T, c: Scalar): T;
    static minimum<T extends NDArray>(a: NDArray, b: NDArray): T;
    static minimumStrict<T extends NDArray>(a: T, b: T): T;
    static maximum<T extends NDArray>(a: NDArray, b: NDArray): T;
    static maximumStrict<T extends NDArray>(a: T, b: T): T;
}
