package gss.act;

import gss.*;
import gss.arr.*;
import java.util.*;

import static gss.Util.*;

public class Softmax extends Activation
{
	// need many improvements on forward method.
	@Override
	public Base forward(Base arr)
	{
		// !!!!!! don't put negative values in softmax layer.

		/*
		 how to calculate?
		 1. calculate the sum of the array.
		 2. calculate exponent value for evenry element.
		 3. devide the exponent value with the sum.
		 !!! then you value will be softmaxes.
		 what is the use of max value in this calculation.
		 it is used for numerical stability. because of the function Math.exp(x).
		 it explodes when the value is large.
		 so what we need is we need to clip the values that means the largest value will be 1.
		 x - max will be < 0. the result will be the same.
		 */
		Base arr2d=arr.as2DArray();
		float[][] arrOut=new float[arr2d.shape[0]][arr2d.shape[1]];
		for (int d=0;d < arr2d.shape[0];d++)
		{
			// float[] ar=arr2d[d];
			int ar=arr2d.shape[1];
			// Find the maximum value in the input vector
			double max = arr2d.get(d, 0);
			for (int i = 1; i < arr2d.shape[1]; i++)
			{
				if (arr2d.get(d, i) > max)
				{
					max = arr2d.get(d, i);
				}
			}

			// Calculate exponentials
			float sum=0;
			float[] exponentials = new float[ar];
			for (int i = 0; i < ar; i++)
			{
				float exps=(float)Math.exp(arr2d.get(d, i) - max);
				exponentials[i] = exps;
				sum += exps;
			}
			// Calculate softmax probabilities
			float[] probabilities = new float[ar];
			for (int i = 0; i < ar; i++)
			{
				probabilities[i] = exponentials[i] / sum;
			}
			arrOut[d] = probabilities;
		}
		Base out= new Base(flatten(arrOut)).reshape(arr.shape); // .setEnableGradient(arr.requiresGradient());
		// gradient calculator in progress.
		// out.setGradientFunction(softmaxGradient, arr);
		return out;
	}
	public static float[] softmaxForward(float[] x)
	{
		int n = x.length;
		float[] result = new float[n];

		// Numerical stability: subtract max(x) to avoid overflow
		float max = x[0];
		for (float val : x)
		{
			if (val > max) max = val;
		}

		// Compute exponentials and sum
		float sum = 0.0f;
		for (int i = 0; i < n; i++)
		{
			result[i] = (float) Math.exp(x[i] - max); // Shift by max
			sum += result[i];
		}

		// Normalize
		for (int i = 0; i < n; i++)
		{
			result[i] /= sum;
		}
		return result;
	}
	public static float[] softmaxBackward(float[] grad, float[] x)
	{
		int n = x.length;
		float[] softmax = softmaxForward(x); // Recompute softmax (no caching)
		float[] gradInput = new float[n];

		// Compute dot product of grad and softmax (gradᵀ ⋅ softmax)
		float dot = 0.0f;
		for (int i = 0; i < n; i++)
		{
			dot += grad[i] * softmax[i];
		}

		// Compute gradient for each element: gradInput_i = softmax_i * (grad_i - dot)
		for (int i = 0; i < n; i++)
		{
			gradInput[i] = softmax[i] * (grad[i] - dot);
		}

		return gradInput;
	}
//	public static GradFunc softmaxGradient=new GradFunc("softmax"){
//		@Override
//		public NDArray backward(NDArray host, NDArray[] childs, Object[] params)
//		{
//			float[] grd=host.base.toArray();
//			float[] dt=childs[0].base.toArray();
//			float[]bc=softmaxBackward(grd, dt);
//			// softmax in progress.
//			childs[0].base.data.setGrad(bc); // don't use this method.
//			return null;
//		}
//	};
//	public static void main2(String[] args)
//	{
//		float[] input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
//		NDArray in=new NDArray(input);
//		NDArray out = new Softmax().forward(in);
//		float[] output=out.base.toArray();
//		System.out.println(Arrays.toString(output)); // Example output (values may vary slightly)
//
//	}

}
