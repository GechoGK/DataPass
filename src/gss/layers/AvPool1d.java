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
		o.setRequiresGradient(in.hasGradient());
		o.setGradientFunctionS(avPool1dGradient, poolSize, in);
		// reshape ths output array based on the input array.
		int[]outShape=input.shape.clone();
		outShape[outShape.length - 1] = outSize;
		o = o.reshape(outShape);
		return o;
	}
	public static GradFunc avPool1dGradient=new GradFunc("maxPool1d"){
		@Override
		public Base backward(Base host, Base[] childs, Object params)
		{
			int poolSize=params;
			Base ch = childs[0].detachGradient();
			Base grd = host.detachGradient();
			if (ch.shape.length != 2 || (ch.shape[0] != grd.shape[0]))
				throw new RuntimeException("invalid array length on maxpool1d layer.");
			// a.grad = c.grad * 1 / b.data
			for (int r=0;r < grd.shape[0];r++)
				for (int c=0,ac=0;c < grd.shape[1];c++,ac += poolSize)
				{
					float agrad=grd.get(r, c) * 1 / poolSize;
					for (int p=0;p < poolSize;p++)
						ch.set(new int[]{r, p + ac}, agrad);
				}
			return null;
		}
	};
}
