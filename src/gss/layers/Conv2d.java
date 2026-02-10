package gss.layers;

import gss.*;
import gss.arr.*;
import java.util.*;
import test.*;

import static gss.Util.*;
import static gss.MathUtil.*;

public class Conv2d extends Module
{
	private int inputSize,numChannels,numKernels,kernelSize;
	private int outputSize;
	private boolean useBiase;

	// ... padding and strides.

	public Base biase,kernels;

	public Conv2d(int inputSize, int inputChannels, int outputChannel, int kernelSize)
	{
		this.inputSize = inputSize;
		this.numChannels = inputChannels;
		this.numKernels = outputChannel;
		this.kernelSize = kernelSize;
		useBiase = true;
		init();
	}
	public Conv2d(int inputSize, int inputChannels, int outputChannel, int kernelSize, boolean haveBiase)
	{
		this.inputSize = inputSize;
		this.numChannels = inputChannels;
		this.numKernels = outputChannel;
		this.kernelSize = kernelSize;
		this.useBiase = haveBiase;
		init();
	}
	private void init()
	{
		outputSize = (inputSize - kernelSize) + 1; // for normal convolution.
		int[]sh={numKernels, numChannels, kernelSize, kernelSize};
		kernels = newParam(NDArray.arange(1, length(sh) + 1)).reshapeLocal(sh); // change NDArray.ones -> NDArray.rand
		if (useBiase)
			biase = newParam(NDArray.ones(numKernels, outputSize, outputSize)); // 3d array.
	}
	public void setKernels(Base b)
	{
		if (!Arrays.equals(b.shape, kernels.shape))
		{
			print("can't set to kernel.because they have different shape.");
			return;
		}
		// print("==", b.shape, kernels.shape);
		params.remove(kernels);
		kernels = newParam(b);
	}
	/*
	 a = [[[1,2,3],
	 .     [4,5,6],
	 .     [7,8,9]],
	 .    [[10,11,12],
	 .     [13,14.15]
	 .     [16,17,18]]  = 1,2,3,3 (2 cahnnel 3 x 3) image.
	 b = [[[[1,2],
	 .      [3,4]],
	 .     [[5,6]
	 .      [7,8]],
	 .    [[[9,10],
	 .      [11,12]],
	 .     [[13,14]
	 .      [15,16]]] = 2,2,2,2 the kernel is already flipped.
	 // for kernel 1 channel 1
	 = 1*1 + 2*2 + 4*3 + 5*4 = 37
	 = 2*1 + 3*2 + 5*3 + 6*4 = 47
	 = 4*1 + 5*2 + 7*3 + 8*4 = 67
	 = 5*1 + 6*2 + 8*3 + 9*4 = 77.
	 ---------
	 // for kernel 1 channel 2
	 = 10*5 + 11*6 + 13*7 + 14*8 = 319
	 = 11*5 + 12*6 + 14*7 + 15*8 = 345
	 = 13*5 + 14*6 + 16*7 + 17*8 = 397
	 = 14*5 + 15*6 + 17*7 + 18*8 = 423
	 ----
	 for kernel 2 channel 1
	 = 1*9 + 2*10 + 4*11 + 5*12 = 133
	 = 2*9 + 3*10 + 5*11 + 6*12 = 175
	 = 4*9 + 5*10 + 7*11 + 8*12 = 259
	 = 5*9 + 6*10 + 8*11 + 9*12 = 301
	 ---------
	 = 10*13 + 11*14 + 13*15 + 14*16 = 703
	 = 11*13 + 12*14 + 14*15 + 15*16 = 761
	 = 13*13 + 14*14 + 16*15 + 17*16 = 877
	 = 14*13 + 15*14 + 17*15 + 18*16 = 935

	 ==[[[37,47],   ++ [319,345]
	 .   [67,77],   ++  [397,423]]
	 .  [[133,175], ++ [[703,761]
	 .   [259,301]] ++  [877,935]]]
	 ==========
	 . [[[356, 392],
	 .   [464, 500],
	 .  [[836, 936],
	 .   [1136, 1236]]]  2,2,2
	 ------ gradient ------

	 1,2-----\
	 .      * 1,2 ---\
	 .  1,2--/        \
	 .               + 5,2---\
	 2,4-----\        /       \
	 .      * 4,2 ---/         \
	 .  2,4--/                  \
	 . 						   + 37,2
	 4,6-----\                  /
	 .      * 12,2 --\         /
	 .  3,8--/        \       /
	 .               + 32,2--/
	 5,8-----\       /
	 .      * 20,2--/
	 .  4,10--/
	 -----------
	 input gradient;
	 == 37 ==
	 1 = 1*2 = 2 & 
	 2 = 2*2 = 4
	 4 = 3*2 = 6
	 5 = 4*2 = 8
	 == 47 ==
	 2 = 1*2 = 2 + 4 = 6
	 3 = 2*2 = 4
	 5 = 3*2 = 6 + 8 = 14
	 6 = 4*2 = 8
	 == 67 ==
	 4 = 1*2 = 2 + 6  = 8
	 5 = 2*2 = 4 + 14 = 18
	 7 = 3*2 = 6
	 8 = 4*2 = 8
	 == 77 ==
	 5 = 1*2 = 2 + 18 = 20
	 6 = 2*2 = 4 + 8  = 12
	 8 = 3*2 = 6 + 8  = 14
	 9 = 4*2 = 8
	 +++++++++++
	 == 319 ==
	 10 = 5*2 = 10
	 11 = 6*2 = 12
	 13 = 7*2 = 14
	 14 = 8*2 = 16
	 == 345 ==
	 11 = 5*2 = 10 + 12 = 22
	 12 = 6*2 = 12
	 14 = 7*2 = 14 + 16 = 30
	 15 = 8*2 = 16
	 == 397 ==
	 13 = 5*2 = 10 + 14 = 24
	 14 = 6*2 = 12 + 30 = 42
	 16 = 7*2 = 14
	 17 = 8*2 = 16
	 == 423 ==
	 14 = 5*2 = 10 + 42 = 52
	 15 = 6*2 = 12 + 16 = 28
	 17 = 7*2 = 14 + 16 = 30
	 18 = 8*2 = 16
	 ++++++++++++
	 == 133 ==
	 1 = 9 *2 = 18
	 2 = 10*2 = 20
	 4 = 11*2 = 22
	 5 = 12*2 = 24
	 == 175 ==
	 2 = 9 *2 = 18 + 20 = 38
	 3 = 10*2 = 20
	 5 = 11*2 = 22 + 24 = 46
	 6 = 12*2 = 24
	 == 259 ==
	 4 = 9 *2 = 18 + 22 = 40
	 5 = 10*2 = 20 + 46 = 66
	 7 = 11*2 = 22
	 8 = 12*2 = 24
	 == 301 ==
	 5 = 9 *2 = 18 + 66 = 84
	 6 = 10*2 = 20 + 24 = 44
	 8 = 11*2 = 22 + 24 = 46
	 9 = 12*2 = 24
	 +++++++++++
	 == 703 ==
	 10 = 13*2 = 26
	 11 = 14*2 = 28
	 13 = 15*2 = 30
	 14 = 16*2 = 32
	 == 761 ==
	 11 = 13*2 = 26 + 28 = 54
	 12 = 14*2 = 28
	 14 = 15*2 = 30 + 32 = 62
	 15 = 16*2 = 32
	 == 877 ==
	 13 = 13*2 = 26 + 30 = 56
	 14 = 14*2 = 28 + 62 = 90
	 16 = 15*2 = 30
	 17 = 16*2 = 32
	 == 935 ==
	 14 = 13*2 = 26 + 90 = 116
	 15 = 14*2 = 28 + 32 = 60
	 17 = 15*2 = 30 + 32 = 62
	 18 = 16*2 = 32

	 === [[[[18 + 2, 38 + 6, 20 + 4],
	 .      [40 + 8, 84 + 20, 44 + 12],
	 .      [22 + 6, 46 + 14, 24 + 8]],
	 .     [[26 + 10, 54 + 22, 28 + 12],
	 .      [56 + 24, 116 + 52, 60 + 28],
	 .      [30 + 14, 62 + 30, 32 + 16]]]];
	 === [[[[20, 44,  24],
	 .      [48, 104, 56],
	 .      [28, 60,  32]],
	 .     [[36, 76,  40],
	 .      [80, 168, 88],
	 .      [44, 92,  48]]]]; 1,2,3,3
	 ---- kernel the same way.


	 // multiplication grad
	 a.g = c.g * b
	 b.g = c.g * a

	 c.g=[[[2,3],
	 .     [4,5]],
	 .    [[6,7],
	 .     [8,9]]]. 2,2,2


	 */
	@Override
	public Base forward(Base dataIn)
	{
		// needs testing againest TestValue.
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
		float[][][][]inp=copy4(in);
		float[][][][]krn=copy4(kernels);
		int k_size=kernels.shape[3];
		// output = batch,numKernels,outSize,outSize
		for (int bs=0;bs < in.shape[0];bs++) // loop over batch size
		{
			for (int ks=0;ks < numKernels;ks++) // loop over number of kernels
			{
				for (int cs=0;cs < in.shape[1];cs++) // loop over channels size
				{
					// MathUtil.conv2d(in.slice(bs, cs), kernels.slice(ks, cs), !haveBiase ?null: biase.slice(ks), out[bs][ks]);
					for (int or=0;or < outputSize;or++)
					{
						for (int oc=0;oc < outputSize;oc++) // we used one loop for caching kernel.
						{
							float sm = 0;
							for (int kr=k_size - 1,ir=0;kr >= 0;kr--,ir++)
							{
								for (int kc=k_size - 1,ic=0;kc >= 0;kc--,ic++)
								{
									sm += inp[bs][cs][ir + or][ic + oc] * krn[ks][cs][kr][kc];
								}
							}
							out[bs][ks][or][oc] += sm;
							// out[bs][ks][or][oc] += sm + bis[ks][or][oc];
						}
					}
				}
			}
		}
		Base o=NDArray.wrap(out);
		o.setRequiresGradient(in.hasGradient() || kernels.hasGradient()); // ignore biase, kernel is enough.
		o.setGradientFunctionS(conv2dGradient, in, kernels);
		if (useBiase)
			o = NDArray.add(o, biase);
		return o;
	}
	public static GradFunc conv2dGradient=new GradFunc("conv2d"){
		@Override
		public Base backward(Base host, Base[] childs, Object params)
		{
			Base input=childs[0];
			Base kernel=childs[1];

			float[][][][]in=copy4(input);
			float[][][][]inG=new float[in.length][in[0].length][in[0][0].length][in[0][0][0].length];
			float[][][][]krn=copy4(kernel);
			float[][][][]krnG=new float[krn.length][krn[0].length][krn[0][0].length][krn[0][0][0].length];
			float[][][][]grd=copy4(host.detachGradient());

			for (int b=0;b < input.shape[0];b++) // loop over input batch.
			{
				for (int k=0;k < kernel.shape[0];k++) // loop over number of kernels.
				{
					for (int ch=0;ch < kernel.shape[1];ch++) // loop over imput channels. // ignored for now.					{
					{
						float[][]ins=in[b][ch];
						float[][]krs=krn[k][ch];
						// System.out.println("ccc " + k + ", " + ch);
						agrad(ins, krs, grd[b][k], inG[b][ch], krnG[k][ch]);
					}
				}
			}
			MathUtil.append(kernel.detachGradient(), krnG);
			MathUtil.append(input.detachGradient(), inG);
			return null;
		}
		void agrad(float[][]in, float[][]k, float[][]grd, float[][]aout, float[][]bout)
		{
			for (int gr=0;gr < grd.length;gr++)
				for (int gc=0;gc < grd[0].length;gc++)
				{
					float gval=grd[gr][gc];
					for (int ir=0, kr=k.length - 1;kr >= 0;kr--,ir++)
						for (int ic=0,kc=k[0].length - 1;kc >= 0;kc--,ic++)
						{
							float kval=k[kr][kc];
							float ival=in[gr + ir][gc + ic];
							// ag += kval * gval;
							aout[gr + ir][gc + ic] += kval * gval;
							// kg += ival * gval;
							bout[kr][kc] += ival * gval;
						}
				}
		}
	};
}
