import { Array1D, Array2D, Array3D, Array4D, NDArray } from './ndarray';
import { Rank } from './types';
export declare class Ops {
    static batchNormalization2D(x: Array2D, mean: Array2D | Array1D, variance: Array2D | Array1D, varianceEpsilon?: number, scale?: Array2D | Array1D, offset?: Array2D | Array1D): Array2D;
    static batchNormalization3D(x: Array3D, mean: Array3D | Array1D, variance: Array3D | Array1D, varianceEpsilon?: number, scale?: Array3D | Array1D, offset?: Array3D | Array1D): Array3D;
    static batchNormalization4D(x: Array4D, mean: Array4D | Array1D, variance: Array4D | Array1D, varianceEpsilon?: number, scale?: Array4D | Array1D, offset?: Array4D | Array1D): Array4D;
    static batchNormalization<R extends Rank>(x: NDArray<R>, mean: NDArray<R> | Array1D, variance: NDArray<R> | Array1D, varianceEpsilon?: number, scale?: NDArray<R> | Array1D, offset?: NDArray<R> | Array1D): NDArray<R>;
}
