import { Array1D, Array2D, Array3D, Array4D, NDArray } from './ndarray';
export declare class Ops {
    static concat1D(a: Array1D, b: Array1D): Array1D;
    static concat2D(a: Array2D, b: Array2D, axis: number): Array2D;
    static concat3D(a: Array3D, b: Array3D, axis: number): Array3D;
    static concat4D(a: Array4D, b: Array4D, axis: number): Array4D;
    static concat<T extends NDArray>(a: T, b: T, axis: number): T;
}
