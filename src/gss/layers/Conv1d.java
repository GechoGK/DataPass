package gss.layers;

import gss.*;
import gss.arr.*;
import java.util.*;

import static gss.Util.*;

public class Conv1d extends Module
{
	private int n_channels;
	private int n_kernels;
	private int kernel_size;
	private int output_size=0;
	private int input_size=0;
	
	public Base kernels,biase;

	public Conv1d(int input_size, int n_channels, int n_kernels, int kernel_size)
	{
		this.input_size = input_size;
		this.n_channels = n_channels;
		this.n_kernels = n_kernels;
		this.kernel_size = kernel_size;
		init();
	}
	private void init()
	{
		output_size = input_size - kernel_size + 1; // for normal convolution.
		kernels = newParam(NDArray.ones(n_kernels, n_channels, kernel_size)); // change NDArray.ones -> NDArray.rand
		biase = newParam(NDArray.ones(n_kernels, output_size));
	}
	@Override
	public Base forward(Base input)
	{
		/*
		 how does convolution works on multi dimension array.
		 -- the input data can be any dimension.
		 -- we convert n-dimension array into 2-D array, to make it easier for calculation.
		 -- the kernel is 3-D array.
		 .. 0 - kernels count. e.g  2
		 .. 1 - features(channels). e.g rgb = 3
		 .. 2 - kernel size. e.g 5
		 -- so the final dimension of kernel will be arr[2,3,5].
		 !!! ho can we calculate on these 3 dims.
		 let's say the input dim is 2 [4,10];
		 -- output = new [input.shape[0],kernels.shape[0], outputSize];
		 -- first we iterate through all out inputs outer dim.
		 : for di in input.shape[0];  accessing all our data rows.
		 -- then we loop over our kernels count(outer dim)
		 : for krs in kernels.shape[0]; // iterate through all kernels count.
		 : add float array (karr) to store the sum of features convolved with current input(di) and initialize with 0.
		 -- again loop over kernel features inside the current kernel(krs).	
		 : for kft in kernels.shape[1]; // looping over all the features of kernel.
		 // now we have 1d input(di) and 1d kernel feature (kft)
		 -- then we do the convolution of two and add the result into karr.
		 : karr += convolve( di, kft );
		 -- after all the features (kft) are added into karr.
		 -- we now put karr into output[currIn,currKer]
		 : output[di.i, krs.i ] = karr;
		 -- done. return it.
		 : return output.
		 */
		if (n(input.shape, 0) != input_size)
			throw new IllegalArgumentException("input data size must be equal to input size(" + input_size + ")");
		if ((input.shape.length < 2 && n_channels != 1) | (input.shape.length > 1 && n(input.shape, 1) != n_channels))
			throw new IllegalArgumentException("input data feature size must be equal to features(" + n_channels + ").");
		Base in=input.reshape(-1, n_channels/* n(input.shape, 1)*/, n(input.shape, 0)); // 3d array.
		// float[][][] out=new float[in.shape[0]][n_kernels][output_size];
		float[] outF=new float[in.shape[0] * n_kernels * output_size];
		// input iteration.
		// System.out.println(Arrays.toString(in.shape) + ", " + n_kernels + ", " + n_channels);
		int pos=0;
		for (int din=0;din < in.shape[0];din++) // for input data count.
		{
			// System.out.println("data ::: " + din);
			for (int kr=0;kr < n_kernels;kr++) // loop over number of kernels.
			{
				// System.out.println("kernel ::: " + kr);
				// float[] karr=new float[output_size];
				int tmpPos=pos;
				for (int chn=0;chn < n_channels;chn++) // loop over number of channels.
				{
					// System.out.println("channels :::" + chn);
					// convolved1k1(in.get(din, chn), kernels.get(kr, chn), karr);
					// test area
					pos = tmpPos;
					for (int w=0;w < output_size;w++)
					{
						float sm=0;
						int kp=kernel_size - 1;
						for (int k=0;k < kernel_size;k++)
						{
							float kv=kernels.get(kr, chn, kp--);
							sm += in.get(din, chn, w + k) * kv; // get kernel in reverse order
						}
						// out[din][kr][w] += sm;
						outF[pos++] += sm + biase.get(kr, w);
					}
					// pos++;
				}
				// out[din][kr] = karr;
			}
		}
		Base output=new Base(outF, new int[]{in.shape[0],kernels.shape[0],output_size});
		output.setRequiresGradient(kernels.requiresGradient() | biase.requiresGradient() | input.requiresGradient());
		output.setGradientFunction(conv1dGradient, kernels, biase, input);
		// System.out.println("... " + input + " >>> " + d);
		// needs reshaping.
		// System.out.println("===== equals :" + Arrays.equals(outF, flatten(out)));
		/*
		 example.
		 --- input.shape = {2,3,12} // 3 = fearures size, 12 = inputLength
		 others (2) = input data depth maybe more.
		 but the ouput is always 3d, even if the input is 5d like.
		 --- input.shape = {2,3,2,5,12}
		 --- 12 = input length , 5 = features , 2,3,2 = data depth. an be multiplied.
		 output = 12, n_kernels, inputLength-kernel_size.
		 so what we do is we extract shapes from input array then we apply it into output without touching the last two shapes on the output.
		 */
		int rmLen=Math.max(0, input.shape.length - (output.shape.length - 1));
		int[] sh=new int[rmLen + 2];
		// System.out.println("== " + rmLen);
		int sum=1;
		for (int i=0;i < rmLen;i++)
		{
			sh[i] = input.shape[i];
			sum *= sh[i];
		}
		if (sum != output.shape[0])
			System.out.println("unable to expand the shape from " + Arrays.toString(output.shape) + " to " + Arrays.toString(sh));
		else
		{
			sh[sh.length - 1] = output_size; // last
			sh[sh.length - 2] = kernels.shape[0]; // last -1
			// System.out.println("array reshaped and trimmed from " + Arrays.toString(output.shape) + " to " + Arrays.toString(sh));
			output = output.reshape(sh); // also trimmed.
		}
		return output;
	}
//	float[] convolved1k1(float datain, float krn, float[] out)
//	{
//		// do convolution on 1d level.
//		for (int w=0;w < output_size;w++)
//		{
//			float sm=0;
//			int kp=kernel_size;
//			for (int k=0;k < kernel_size;k++)
//			{
//				float kv=kernel.get(kp--);
//				sm += in.get(0, 0, w + k) * kv; // get kernel in reverse order
//			}
//			out[0][0][w] = sm;
//		}
//		return out;
//	}
	void forwardOld(Base input)
	{
		/*
		 if (n(input.shape, 0) != input_size)
		 throw new IllegalArgumentException("input data size must be equal to input size(" + input_size + ")");
		 if (n(input.shape, 1) != n_channels)
		 throw new IllegalArgumentException("input data feature size must be equal to features(" + n_channels + ").");
		 NDArray[]outs=new NDArray[n_kernels];
		 for (int i=0;i < n_kernels;i++)
		 {
		 NDArray kd=null;
		 for (int c=0;c < n_channels;c++)
		 {
		 NDArray k=kernels.get(i, c);
		 NDArray crs=input.convolve1d(k);
		 kd = kd == null ?crs: kd.add(crs);
		 }
		 outs[i] = kd;
		 }
		 NDArray out=NDArray.merge(outs);
		 // System.out.println(Arrays.toString(out.getShape()) + ", " + Arrays.toString(biase.getShape()));
		 return out.add(biase);
		 */
	}
	public static GradFunc conv1dGradient = new GradFunc("conv1d"){
		@Override
		public Base backward(Base host, Base[] childs, Object params)
		{
			return null;
		}
	};
}
