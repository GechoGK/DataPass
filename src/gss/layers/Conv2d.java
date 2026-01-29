package gss.layers;

import gss.*;
import gss.arr.*;

public class Conv2d extends Module
{
	private int inputSize,numChannels,numKernels,kernelSize;
	private int outputSize;
	private boolean haveBiase;

	private Base biase,kernels;

	public Conv2d(int inputSize, int numChannels, int numKernels, int kernelSize)
	{
		this.inputSize = inputSize;
		this.numChannels = numChannels;
		this.numKernels = numKernels;
		this.kernelSize = kernelSize;
		haveBiase = true;
		init();
	}
	private void init()
	{
		outputSize = (inputSize - kernelSize) + 1; // for normal convolution.
		kernels = newParam(NDArray.rand(numKernels, numChannels, kernelSize, kernelSize)); // change NDArray.ones -> NDArray.rand
		if (haveBiase)
			biase = newParam(NDArray.ones(numKernels, outputSize, outputSize)); // 3d array.
	}
	@Override
	public Base forward(Base dataIn)
	{
		/*
		 1 single kernel size = 2d array. eg (32x32).
		 multiple channel size kernel = 3d array (channels, size,size) = (3,32,32) rgb,input,input;
		 multiple kernels with muliple chanels = 4d array (num kernels,num channels,kernsize,kernsize) = (50,3,32,32);

		 dataIn should be 4d array. (batch, channels, size,size);
		 output array dim = 4d array (batch, num_kernels, size,size);
		 */

		Base in=dataIn.as4DArray(); // 4d array (batch_size, num_channels, input_size, input_size);

		// for single kernel.
		

		return null;
	}
}
