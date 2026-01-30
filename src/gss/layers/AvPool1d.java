package gss.layers;

import gss.*;
import gss.arr.*;

public class AvPool1d extends Module
{
	private int poolSize;

	public AvPool1d(int poolSize)
	{
		this.poolSize = poolSize;
	}

	@Override
	public Base forward(Base input)
	{
		Base in=input.as2DArray();
		int outSize=in.shape[1] / poolSize;
		float[][]out=new float[in.shape[0]][outSize];
		for (int b=0;b < in.shape[0];b++)
		{
			MathUtil.averagePool1d(in.slice(b), poolSize, out[b]);
		}
		Base o=NDArray.wrap(out);
		o.setRequiresGradient(input.hasGradient());
		// o.setGradientFunctionS(avPool1dGradient, index, input);
		// reshape ths output array based on the input array.
		int[]outShape=input.shape.clone();
		outShape[outShape.length - 1] = outSize;
		o = o.reshape(outShape);
		return o;
	}

}
