package gss.layers;

import gss.*;
import gss.arr.*;

import static gss.Util.*;

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
		// check input size.
		Base in=dataIn.as4DArray(); // 4d array (batch_size, num_channels, input_size, input_size);
		if (in.shape[1] != numChannels)
			error("number of channels for the input must be \"" + numChannels + "\"");
		// output = batch_size, 
		float[][][][] out=new float[in.shape[0]][numKernels][outputSize][outputSize];
		// output = batch,numKernels,outSize,outSize
		for (int bs=0;bs < in.shape[0];bs++) // loop over batch size
		{
			for (int ks=0;ks < numKernels;ks++) // loop over number of kernels
			{
				for (int cs=0;cs < in.shape[1];cs++) // loop over channels size
				{
					MathUtil.conv2d(in.slice(bs, cs), kernels.slice(ks, cs), out[bs][ks]);
				}
			}
		}
		Base o=NDArray.wrap(out);
		return o;
	}
}
