package gss;

import gss.arr.*;

import static gss.Util.*;

public class MathUtil
{
	public static float[]conv1d(Base in, Base kern)
	{
		return conv1d(in, kern, null);
	}
	public static float[]conv1d(Base in, Base kern, float[]out)
	{
		// good optimization technique.
		if (in.getDim() != 1 || kern.getDim() != 1)
			throw new RuntimeException("convolve 1d error : expected 1d kernel array and 1d data.");

		int klen=kern.length;
		int len=in.length - klen + 1;
		if (out == null)
			out = new float[len];
		if (out.length != len)
			error("the parameter out array length should be " + len + ", found :" + out.length);
		float[]kcache=new float[klen];
		float[]incache=new float[in.length];
		// caching the kernel, and the portion of data.
		float sm=0;
		int p=0;
		for (int k=klen - 1;k >= 0;k--)
		{
			float kv=kern.get(k);
			float iv=in.get(p);
			kcache[p] = kv;
			incache[p] = iv;
			sm += iv * kv;
			p++;
		}
		out[0] += sm;
		// kernel cached use cache kernel, to gain some performance.
		for (int i=1;i < len;i++) // starts from 1, 0 used for kernel caching...
		{
			int k=klen - 1;
			float iv=in.get(i + k);
			incache[i + k] = iv;
			sm = iv * kcache[k];
			k--;
			for (;k >= 0;k--)
			{
				sm += incache[i + k] * kcache[k];
			}
			out[i] += sm;
		}
		return out;
	}
	public static float[][] conv2d(Base inp, Base krn)
	{
		if (krn.getDim() != 2 || krn.shape[0] != krn.shape[1])
			error("the kernel dimension should be 2: found = " + krn.getDim());
		if (inp.getDim() != 2)
			error("input dimension needs to be 2, found :" + inp.getDim());

		int k_size=krn.shape[0];
		int outputR=(inp.shape[0] - krn.shape[0]) + 1;
		int outputC=(inp.shape[1] - k_size) + 1;

		float[][] in=copy(inp);
		float[][] kern=copy(krn);

		float[][] out=new float[outputR][outputC];

		for (int or=0;or < outputR;or++)
		{
			for (int oc=0;oc < outputC;oc++) // we used one loop for caching kernel.
			{
				float sm = 0;
				for (int kr=k_size - 1,ir=0;kr >= 0;kr--,ir++)
				{
					for (int kc=k_size - 1,ic=0;kc >= 0;kc--,ic++)
					{
						sm += in[ir + or][ic + oc] * kern[kr][kc];
					}
				}
				out[or][oc] = sm;
			}
		}
		return out;
	}
	public static float[][]copy(Base in)
	{
		if (in.getDim() != 2)
			error("the input dimension must be 2, found :" + in.getDim());
		float[][] o=new float[in.shape[0]][in.shape[1]];
		for (int or=0;or < in.shape[0];or++)
			for (int oc=0;oc < in.shape[1];oc++) // we used one loop for caching kernel.
				o[or][oc] = in.get(or, oc);
		return o;
	}
}
