package gss.layers;

import gss.*;
import gss.arr.*;
import java.util.*;

import static gss.Util.*;
import static gss.MathUtil.*;
import java.util.stream.*;

public class Conv1d extends Module
{
	public int n_channels; // or features
	public int n_kernels;
	public int kernel_size;
	public int output_size=0;
	public int input_size=0;

	private boolean haveBiase=false;

	public Base kernels,biase;

	// ... padding and strides

	/*
	 input dimension = (batch_size, feature_size, input_size);
	 kernel dimension = (num_kernels, num_channels, kernel_size);
	 biase dimension = (num_kernels, output_size);

	 output_size = (input_size - kernel_size) + 1;
	 */

	public Conv1d(int input_size, int n_channels, int n_kernels, int kernel_size)
	{
		this(input_size, n_channels, n_kernels, kernel_size, true);
	}
	public Conv1d(int input_size, int n_channels, int n_kernels, int kernel_size, boolean useBiase)
	{
		this.input_size = input_size;
		this.n_channels = n_channels;
		this.n_kernels = n_kernels;
		this.kernel_size = kernel_size;
		haveBiase = useBiase;
		init();
	}
	private void init()
	{
		output_size = (input_size - kernel_size) + 1; // for normal convolution.
		kernels = newParam(NDArray.rand(n_kernels, n_channels, kernel_size)); // change NDArray.ones -> NDArray.rand
		if (haveBiase)
			biase = newParam(NDArray.ones(n_kernels, output_size));
	}
	@Override
	public Base forward(Base input)
	{
		if (n(input.shape, 0) != input_size)
			throw new IllegalArgumentException("input data size must be equal to input size(" + input_size + ")");
		if ((input.shape.length < 2 && n_channels != 1) | (input.shape.length > 1 && n(input.shape, 1) != n_channels))
			throw new IllegalArgumentException("input data feature size must be equal to features(" + n_channels + ").");
		Base inp=input.reshape(-1, n_channels/* n(input.shape, 1)*/, n(input.shape, 0)); // 3d array(batch_size, channel, input_size)
		// float[][][] out=new float[in.shape[0]][n_kernels][output_size];
		float[] outF=new float[inp.shape[0] * n_kernels * output_size];
		// input iteration.
		// System.out.println(Arrays.toString(in.shape) + ", " + n_kernels + ", " + n_channels);
		float[][][] in=copy3(inp);
		float[][][]krn=copy3(kernels);
		int pos=0;
		for (int din=0;din < inp.shape[0];din++) // batch_size
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
					for (int w=0;w < output_size;w++) // increment by inc. default = 1.
					{
						float sm=0;
						int kp=kernel_size - 1;
						for (int k=0;k < kernel_size;k++)
						{
							float kv=krn[kr][chn][kp];
							sm += in[din][chn][w + k] * kv; // get kernel in reverse order
							kp--;
						}
						outF[pos++] += sm;						
					}
				}			
			}
		}
		// System.out.println(sb.toString());
		Base output=new Base(outF, new int[]{inp.shape[0],kernels.shape[0],output_size}); // output=(batch_size, number_of_kernels, output_size);
		output.setRequiresGradient(kernels.hasGradient() | biase.hasGradient() | inp.hasGradient());
		output.setGradientFunctionS(conv1dGradient, kernels, biase, inp);

		if (haveBiase)
			output = NDArray.add(output, biase);
		/*
		 // System.out.println("... " + input + " >>> " + d);
		 // needs reshaping.
		 // System.out.println("===== equals :" + Arrays.equals(outF, flatten(out)));
		 // /*
		 example.
		 --- input.shape = {2,3,12} // 3 = fearures size, 12 = inputLength
		 others (2) = input data depth maybe more.
		 but the ouput is always 3d, even if the input is 5d like.
		 --- input.shape = {2,3,2,5,12}
		 --- 12 = input length , 5 = features , 2,3,2 = data depth. and multiplied.
		 output = 12, n_kernels, inputLength-kernel_size.
		 so what we do is we extract shapes from input array then we apply it into output without touching the last two shapes on the output.
		 //*
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
		 System.out.println("unable to expand the shape from " + Arrays.toString(output.shape) + " to " + Arrays.toString(sh) + " :: ignored");
		 else
		 {
		 sh[sh.length - 1] = output_size; // last
		 sh[sh.length - 2] = kernels.shape[0]; // last -1
		 // System.out.println("array reshaped and trimmed from " + Arrays.toString(output.shape) + " to " + Arrays.toString(sh));
		 output = output.reshape(sh); // also trimmed.
		 }
		 */
		return output;
	}
	public static GradFunc conv1dGradient = new GradFunc("conv1d"){
		@Override
		public Base backward(Base host, Base[] childs, Object params)
		{
			Base kern=childs[0]; // kernel 3d array.
			// Base biase=childs[1]; // biase 2d array.
			Base in=childs[2]; // input 3d array.
			// Base host = 3d array.

			for (int dl=0;dl < in.shape[0];dl++)
				for (int kr=0;kr < kern.shape[0];kr++)
				{
					for (int kf=0;kf < kern.shape[1];kf++)
					{
				 		if (in.hasGradient())
						{			
							// set grad for input
							//input[dl][kf].grad = host[kr][kf].grad @f kern[kr][kf] // full correlation.
							//  1d array         1d array               1d array
							// fullCorrelation(host.slice(kr, kf).detachGradient(), kern.slice(kr, kf), in.slice(dl, kf));
							for (int i=0;i < in.shape[2];i++) // increment by inc. default = 1.
							{
								float sm=0;
								int ips=(i - kern.shape[2]) + 1;
								for (int k=0;k < kern.shape[2];k++)
								{
									float kv=kern.get(kr, kf, k); //  kernels.get(kr, chn, kp);
									int kp=ips + k;
									if (kp >= 0 && kp < host.shape[2])
										sm += host.getGrad(dl, kr, kp) * kv;
								}
								in.setGrad(ar(dl, kf, i), sm);
							}
						}
						// --------------------------------------------
						// set grad for kernel
						// kern[kr][kf].grad = host[kr][kf].grad @ input[dl][kf] // valid convolution.
						//   1d array           1d array            1d array
						// convolution(host.slice(kr, kf).detachGradient(), in.slice(dl, kf), kern.slice(kr, kf));
						// convolution mode="valid"
						for (int i=0;i < kern.shape[2];i++) // increment by inc. default = 1.
						{
							float sm=0;
							int kpos=in.shape[2] - 1;
							for (int k=0;k < in.shape[2];k++)
							{
								float kv=in.get(dl, kf, kpos); //  kernels.get(kr, chn, kp);
								sm += host.getGrad(dl, kr, i + k) * kv; //  in.get(din, chn, w + k) * kv; // get kernel in reverse order
								kpos--;
							}
							// out[din][kr][w] += sm;
							kern.setGrad(ar(kr, kf, i), sm); // + biase.get(i));
						}
					}
					// --------------------------------------------
				}
			return null;
		}
	};

}
