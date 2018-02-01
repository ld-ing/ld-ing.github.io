/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {PolymerElement, PolymerHTMLElement} from '../polymer-spec';
import {ActivationFunction, CPPN} from './cppn';

// const CANVAS_UPSCALE_FACTOR = 1;
const MAT_WIDTH = 30;
// Standard deviations for gaussian weight initialization.
const WEIGHTS_STDEV = .6;

// tslint:disable-next-line:variable-name
const NNArtPolymer: new () => PolymerHTMLElement = PolymerElement({
  is: 'nn-art',
  properties: {
    activationFunctionNames: Array,
    selectedActivationFunctionName: String
  }
});

class NNArt extends NNArtPolymer {
  // Polymer properties.
  activationFunctionNames: ActivationFunction[];
  selectedActivationFunctionName: ActivationFunction;

  private cppn: CPPN;

  private inferenceCanvas: HTMLCanvasElement;

  private z1Scale: number;
  private z2Scale: number;
  private numLayers: number;

  ready() {
    this.inferenceCanvas =
        this.querySelector('#inference') as HTMLCanvasElement;

    this.inferenceCanvas.style.width = `${window.innerWidth}px`;
    this.inferenceCanvas.style.height = `${Math.round(window.innerHeight/2)}px`;

    this.cppn = new CPPN(this.inferenceCanvas);

    //this.inferenceCanvas.style.width =
    //    `${this.inferenceCanvas.width * 6 * CANVAS_UPSCALE_FACTOR}px`;
    //this.inferenceCanvas.style.height =
    //    `${this.inferenceCanvas.height * 2 * CANVAS_UPSCALE_FACTOR}px`;

    this.activationFunctionNames = ['tanh', 'sin', 'relu', 'step'];
    this.selectedActivationFunctionName = 'tanh';
    this.cppn.setActivationFunction(this.selectedActivationFunctionName);
    this.querySelector('#activation-function-dropdown')!.addEventListener(
        // tslint:disable-next-line:no-any
        'iron-activate', (event: any) => {
          this.selectedActivationFunctionName = event.detail.selected;
          this.cppn.setActivationFunction(this.selectedActivationFunctionName);
        });

    const layersSlider =
        this.querySelector('#layers-slider') as HTMLInputElement;
    const layersCountElement =
        this.querySelector('#layers-count') as HTMLDivElement;
    layersSlider.addEventListener('immediate-value-changed', (event) => {
      // tslint:disable-next-line:no-any
      this.numLayers = parseInt((event as any).target.immediateValue, 10);
      layersCountElement.innerText = this.numLayers.toString();
      this.cppn.setNumLayers(this.numLayers);
    });
    this.numLayers = parseInt(layersSlider.value, 10);
    layersCountElement.innerText = this.numLayers.toString();
    this.cppn.setNumLayers(this.numLayers);

    this.z1Scale = 10;
    this.cppn.setZ1Scale(convertZScale(this.z1Scale));

    this.z2Scale = 10;
    this.cppn.setZ2Scale(convertZScale(this.z2Scale));

    //const randomizeButton = this.querySelector('#random')
    // as HTMLButtonElement;
    //randomizeButton.addEventListener('click', () => {
      //this.cppn.generateWeights(MAT_WIDTH, WEIGHTS_STDEV);
    //});

    this.cppn.generateWeights(MAT_WIDTH, WEIGHTS_STDEV);
    this.cppn.start();
  }
}

function convertZScale(z: number): number {
  return (103 - z);
}

document.registerElement(NNArt.prototype.is, NNArt);
