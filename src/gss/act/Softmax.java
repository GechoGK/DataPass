package gss.act;

import gss.*;
import gss.arr.*;
import java.util.*;

import static gss.Util.*;

public class Softmax extends Activation
{
	// need many improvements on forward method.
	// Base.set(i,...);
	// Base.get(i);
	// the above two methods have to be used properly.
	// not only in this class, but also on other loss functions.
	@Override
	public Base forward(Base array)
	{
		// !!!!!! don't put negative values in softmax layer.

		/*
		 how to calculate?
		 1. calculate the sum of the array.
		 2. calculate exponent value for every element.
		 3. devide the exponent value with the sum.
		 !!! then you value will be softmaxes.
		 what is the use of max value in this calculation.
		 it is used for numerical stability. because of the function Math.exp(x).
		 it explodes when the value is large.
		 so what we need is we need to clip the values that means the largest value will be 1.
		 x - max will be < 0. the result will be the same.
		 */
		Base arr1d=array.as1DArray();
		Base out=softmaxForward(arr1d);

		out.reshape(array.shape).setRequiresGradient(arr1d.hasGradient());
		// gradient calculator in progress.
		if (out.hasGradient())
			out.setGradientFunction(softmaxGradient, arr1d);
		return out;
	}
	public static Base softmaxForward(Base arr)
	{
		float[] arrOut=new float[arr.shape[0]];
		// for (int d=0;d < arr2d.shape[0];d++)
		// {
		// float[] ar=arr2d[d];
		int ar=arr.shape[0];
		// Find the maximum value in the input vector
		double max = arr.getRaw(0);
		for (int i = 1; i < arr.shape[0]; i++)
		{
			if (arr.get(i) > max)
			{
				max = arr.getRaw(i);
			}
		}

		// Calculate exponentials
		float sum=0;
		float[] exponentials = new float[ar];
		for (int i = 0;i < ar; i++)
		{
			float exps=(float)Math.exp(arr.getRaw(i) - max);
			exponentials[i] = exps;
			sum += exps;
		}
		// Calculate softmax probabilities
		float[] probabilities = new float[ar];
		for (int i = 0; i < ar; i++)
		{
			probabilities[i] = exponentials[i] / sum;
		}
		arrOut = probabilities;
		Base out = new Base(arrOut);
		return out;
	}
	public static Base softmaxBackward(Base grad, Base x)
	{
		int n = x.length;
		Base softmax = softmaxForward(x); // Recompute softmax (no caching)
		// float[] gradInput = new float[n];

		// Compute dot product of grad and softmax (gradᵀ ⋅ softmax)
		float dot = 0.0f;
		for (int i = 0; i < n; i++)
		{
			dot += grad.getRaw(i) * softmax.getRaw(i);
		}

		// Compute gradient for each element: gradInput_i = softmax_i * (grad_i - dot)
		for (int i = 0; i < n; i++)
		{
			x.setRawGrad(i, softmax.getRaw(i) * (grad.getRaw(i) - dot));
		}

		return x;
	}
	public static GradFunc softmaxGradient=new GradFunc("softmax"){
		@Override
		public Base backward(Base host, Base[] childs, Object params)
		{
			// float[] grd=host.base.toArray();
			// float[] dt=childs[0].base.toArray();
			Base bc=softmaxBackward(host, childs[0]);
			// softmax in progress.
			// childs[0].base.data.setGrad(bc); // don't use this method.
			return null;
		}
	};
//	public static void main(String[] args)
//	{
//		float[] input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
//		Base in=new Base(input).setRequiresGradient(true);
//		Base out = new Softmax().forward(in);
//		System.out.println(out);
//		out.printArray();
//		System.out.println(line(30));
//		Base o=NDArray.add(out, NDArray.zeros(1));
//		System.out.println(o);
//		o.printArray();
//		System.out.println(line(10));
//		o.fillGrad(1);
//		o.backward();
//		System.out.println(in.detachGradient());
//		in.detachGradient().printArray();
//
//
//	}

}
