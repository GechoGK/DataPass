package gss.layers;

import gss.*;
import gss.arr.*;

import static gss.Util.*;

public class MaxPool2d extends Module
{
	private int[] poolSize;
	private int[][][][]index;

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
		index = new int[out.length][tmpShape[0]][tmpShape[1]][];
		for (int b=0;b < in.shape[0];b++) // loop over evedy batch.
		{
			MathUtil.maxPool2d(in.slice(b), out[b], index[b], poolSize);
		}
		Base o=NDArray.wrap(out);
		o.setRequiresGradient(in.hasGradient());
		o.setGradientFunctionS(maxPool2dGradient, index, in);
		// reshape ths output array based on the input array.
		int[]outShape=input.shape.clone();
		outShape[outShape.length - 1] = tmpShape[1];
		if (outShape.length >= 2)
			outShape[outShape.length - 2] = tmpShape[0];
		o = o.reshape(outShape);
		return o;
	}
	public static GradFunc maxPool2dGradient=new GradFunc("maxPool1d"){
		@Override
		public Base backward(Base host, Base[] childs, Object params)
		{
			int[][][][] index=(int[][][][])params; // batch, output_size.
			Base ch = childs[0].detachGradient();
			Base grd = host.detachGradient();
			if (ch.shape.length != 3 || (ch.shape[0] != index.length))
				throw new RuntimeException("invalid array length on maxpool1d layer.");
			// float[] grd=host.base.data.getGrads();
			for (int b=0;b < grd.shape[0];b++) // batch
				for (int r=0;r < grd.shape[1];r++) // row.
					for (int c=0;c < grd.shape[2];c++) // col
					{
						int[]rc=index[b][r][c];
						ch.set(new int[]{b,rc[0],rc[1]}, grd.get(b, r, c));
					}
			return null;
		}
	};
}
