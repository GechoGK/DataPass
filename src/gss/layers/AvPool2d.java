package gss.layers;

import gss.*;
import gss.arr.*;

public class AvPool2d extends Module
{
	private int[] poolSize;

	public AvPool2d(int pool_size)
	{
		this(pool_size, pool_size);
	}
	public AvPool2d(int...pool_size)
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
			MathUtil.averagePool2d(in.slice(b), out[b], poolSize);
		}
		Base o=NDArray.wrap(out);
		o.setRequiresGradient(in.hasGradient());
		o.setGradientFunctionS(avPool2dGradient, poolSize, in);
		// reshape ths output array based on the input array.
		int[]outShape=input.shape.clone();
		outShape[outShape.length - 1] = tmpShape[1];
		if (outShape.length >= 2)
			outShape[outShape.length - 2] = tmpShape[0];
		o = o.reshape(outShape);
		return o;
	}
	public static GradFunc avPool2dGradient=new GradFunc("maxPool1d"){
		@Override
		public Base backward(Base host, Base[] childs, Object params)
		{
			int[] poolSize=(int[])params; // batch, output_size.
			Base ch = childs[0].detachGradient();
			Base grd = host.detachGradient();
			if (ch.shape.length != 3 || (ch.shape[0] != grd.shape[0]))
				throw new RuntimeException("invalid array length on maxpool1d layer.");
			// a.grad = c.grad * 1 / b.data
			int len=poolSize[0] * poolSize[1];
			for (int b=0;b < grd.shape[0];b++) // batch
				for (int r=0,ar=0;r < grd.shape[1];r++,ar += poolSize[0]) // row.
					for (int c=0,ac=0;c < grd.shape[2];c++,ac += poolSize[1]) // col
					{
						float cgrad=grd.get(b, r, c);
						float agrad=cgrad * 1 / len;
						for (int pr=0;pr < poolSize[0];pr++)
							for (int pc=0;pc < poolSize[1];pc++)
							{
								// int[]rc=index[b][r][c];
								ch.set(new int[]{b,pr + ar,pc + ac}, agrad);
							}
					}
			return null;
		}
	};
}
