package gss.layers;

import gss.*;
import gss.arr.*;

public class MaxPool1d extends Module
{
	private int poolSize;
	public int[][] index;

	public MaxPool1d(int poolSize)
	{
		this.poolSize = poolSize;
	}
	@Override
	public Base forward(Base input)
	{
		Base in=input.as2DArray();
		int outSize=in.shape[1] / poolSize;
		float[][]out=new float[in.shape[0]][outSize];
		index = new int[out.length][outSize];
		for (int b=0;b < in.shape[0];b++)
		{
			MathUtil.maxPool1d(in.slice(b), poolSize, out[b], index[b]);
		}
		Base o=NDArray.wrap(out);
		o.setRequiresGradient(in.hasGradient());
		o.setGradientFunctionS(maxPool1dGradient, index, in);
		// reshape ths output array based on the input array.
		int[]outShape=input.shape.clone();
		outShape[outShape.length - 1] = outSize;
		o = o.reshape(outShape);
		return o;
	}
	public static GradFunc maxPool1dGradient=new GradFunc("maxPool1d"){
		@Override
		public Base backward(Base host, Base[] childs, Object params)
		{
			int[][] index=(int[][])params; // batch, output_size.
			Base ch = childs[0].detachGradient();
			Base grd = host.detachGradient();
			if (ch.shape.length != 2 || (ch.shape[0] != index.length))
				throw new RuntimeException("invalid array length on maxpool1d layer.");
			// float[] grd=host.base.data.getGrads();
			for (int r=0;r < grd.shape[0];r++)
				for (int c=0;c < grd.shape[1];c++)
				{
					ch.set(new int[]{r,index[r][c]}, grd.get(r, c));
				}
			return null;
		}
	};
}
