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
		int mds=in.shape[1] % poolSize;
		if (mds != 0)
			throw new RuntimeException("unable to make maxPool! make sure the pool size is divisble by the total length of the array.");
		int newLen=in.shape[1] / poolSize;
		Base outData=new Base(in.shape[0], newLen);
		index = new int[in.shape[0]][outData.shape[1]];
		for (int r=0;r < in.shape[0];r++) // row count.
			for (int c=0;c < newLen;c++) // column count.
			{
				int np=c * poolSize;
				float pmx=-Float.MIN_VALUE;
				for (int p=np;p < np + poolSize;p++)
				{
					float fp=in.get(r, p);
					if (fp >= pmx)
					{
						pmx = fp;
						index[r][c] = p;
					}
				}
				outData.set(new int[]{r,c}, pmx);
			}
		int[] nsh=input.shape.clone();
		nsh[nsh.length - 1] = newLen; // nsh[nsh.length - 1] / poolSize;
		// outData.reshapeLocal(nsh);
		outData.setRequiresGradient(input.hasGradient());
		outData.setGradientFunction(maxPool1dGradient, input);
		outData.setGradientParams(index);
		return outData.reshape(nsh);
	}
	public static GradFunc maxPool1dGradient=new GradFunc("maxPool1d"){
		@Override
		public Base backward(Base host, Base[] childs, Object params)
		{
			int[][] index=(int[][])params;
			Base ch = childs[0].detachGradient();
			Base grd = host.detachGradient();
			if (ch.shape.length != 2 || (ch.shape[0] == index.length && ch.shape[1] == index[0].length))
				throw new RuntimeException("invalid array length on maxpool1d layer.");
			// float[] grd=host.base.data.getGrads();
			for (int r=0;r < index.length;r++)
				for (int c=0;c < index[r].length;c++)
				{
					ch.set(new int[]{r,index[r][c]}, grd.get(new int[]{r,index[r][c]}));
				}
			return null;
		}
	};
}
