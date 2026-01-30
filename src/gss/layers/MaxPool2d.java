package gss.layers;

import gss.*;
import gss.arr.*;
import static gss.Util.*;

public class MaxPool2d extends Module
{
	private int[] poolSize;

	public MaxPool2d(int pool_size)
	{
		this(pool_size, pool_size);
	}
	public MaxPool2d(int...pool_size)
	{
		this.poolSize = pool_size;
	}
	@Override
	public Base forward(Base input)
	{
		// input shape = (batch, size_row, size_column)
		Base in=input.as3DArray();
		int tmpShape[]={in.shape[1] / poolSize[0], in.shape[2] / poolSize[1]};
		float[][][]out=new float[in.shape[0]][tmpShape[0]][tmpShape[1]];
		for (int b=0;b < in.shape[0];b++) // loop over evedy batch.
		{
			MathUtil.maxPool2d(in.slice(b), out[b], poolSize);
		}
		Base o=NDArray.wrap(out);
		o.setRequiresGradient(input.hasGradient());
		// o.setGradientFunctionS(maxPool2dGradient, index, input);
		// reshape ths output array based on the input array.
		int[]outShape=input.shape.clone();
		outShape[outShape.length - 1] = tmpShape[1];
		if (outShape.length >= 2)
			outShape[outShape.length - 2] = tmpShape[0];
		o = o.reshape(outShape);
		return o;
	}
}
