package gss.layers;

import gss.*;
import gss.arr.*;

public class MaxPool1d extends Module
{
	private int poolSize;

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
		// int[] index=new int[outArr.length];
		for (int r=0;r < in.shape[0];r++)
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
						// index[n] = p;
					}
				}
				outData.set(new int[]{r,c}, pmx);
			}
		int[] nsh=input.shape.clone();
		nsh[nsh.length - 1] = newLen; // nsh[nsh.length - 1] / poolSize;
		outData.reshapeLocal(nsh);
		// out.setGradientFunction(maxPool1dGradient, input).setGradientParams(index);
		return outData;
	}
//	public static GradFunc maxPool1dGradient=new GradFunc("maxPool1d"){
//		@Override
//		public NDArray backward(NDArray host, NDArray[] childs, Object[] params)
//		{
//			int[] index=(int[])params[0];
//			NDArray ch=childs[0];
//			float[] grd=host.base.data.getGrads();
//			for (int i=0;i < grd.length;i++)
//				ch.base.data.setGrad(index[i], grd[i]);
//			return null;
//		}
//	};
}
