import { Array1D, Array2D, Scalar } from './ndarray';
export interface LSTMCell {
    (data: Array2D, c: Array2D, h: Array2D): [Array2D, Array2D];
}
export declare class Ops {
    static multiRNNCell(lstmCells: LSTMCell[], data: Array2D, c: Array2D[], h: Array2D[]): [Array2D[], Array2D[]];
    static basicLSTMCell(forgetBias: Scalar, lstmKernel: Array2D, lstmBias: Array1D, data: Array2D, c: Array2D, h: Array2D): [Array2D, Array2D];
}
