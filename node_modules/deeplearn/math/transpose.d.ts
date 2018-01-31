import { NDArray } from './ndarray';
import { Rank } from './types';
export declare class Ops {
    static transpose<R extends Rank>(x: NDArray<R>, perm?: number[]): NDArray<R>;
}
