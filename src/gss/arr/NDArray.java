package gss.arr;

import gss.*;
import java.util.*;

import static gss.Util.*;
import static gss.arr.GradFunc.*;

public class NDArray
{
	/*
	 -- remove all setRaw functions.
	 */
	public static Base arange(float end)
	{
		return arange(0, end, 1);
	}
	public static Base arange(float str, float end)
	{
		return arange(str, end, 1);
	}
	public static Base arange(float str, float end, float inc)
	{
		float[] f=range(str, end, inc);
		Base ar=new Base(f);
		return ar;
	}
	public static Base ones(int...shape)
	{
		return wrap(1, shape);
	}
	public static Base zeros(int...shape)
	{
		return wrap(0, shape);
	}
	public static Base idt(int size)
	{
		int[] sh={size,size};
		float[]f=new float[length(sh)];
		for (int i=0;i < size;i++)
			f[shapeToIndex(ar(size, size), sh)] = 1;
		Base d=new Base(f, sh);
		return d;
	}
	public static Base wrap(float[]v, int...sh)
	{
		return new Base(v, sh);
	}
	public static Base wrap(float v, int...shape)
	{
		int len=length(shape);
		float[] f=new float[len];
		Arrays.fill(f, v);
		return new Base(f, shape);
	}
	public static Base rand(int...shape)
	{
		return rand(shape, -1); // change -1 into another to use seed value.
	}
	public static Base rand(int[]shape, int seed)
	{
		float[] f=new float[length(shape)];
		Random r=null;
		if (seed != -1)
			r = new Random(seed);
		else
			r = new Random();
		for (int i=0;i < f.length;i++)
			f[i] = r.nextFloat();
		Base d=new Base(f, shape);
		return d;
	}
	// mathematical operations.
	// addition
	public static Base add(Base d1, Base d2)
	{
		int[] sh=getCommonShape(d1.shape, d2.shape);
		Base res=new Base(sh).setRequiresGradient(d1.requiresGradient() | d2.requiresGradient());
		if (res.requiresGradient())
			res.setGradientFunction(additionGradient, d1, d2);
		int len=res.length; // length of the array.
		int[] tmpSh=new int[sh.length]; // temporary shape holder.
		for (int i=0;i < len;i++)
		{
			indexToShape(i, sh, tmpSh);
			float op1=d1.get(tmpSh);
			float op2=d2.get(tmpSh);
			res.set(tmpSh, op1 + op2);
		}
		return res;
	}
	public static Base add(Base d1, float d2)
	{
		int[] sh=d1.shape;
		Base res=new Base(sh).setRequiresGradient(d1.requiresGradient());
		Base data2=new Base(new float[]{d2}); // don't use for computation, it is just for gradient.
		if (res.requiresGradient())
			res.setGradientFunction(additionGradient, d1, data2);
		int len=res.length; // length of the array.
		int[] tmpSh=new int[sh.length]; // temporary shape holder.
		for (int i=0;i < len;i++)
		{
			indexToShape(i, sh, tmpSh);
			float op1=d1.get(tmpSh);
			res.set(tmpSh, op1 + d2);
		}
		return res;
	}
	// subtraction
	public static Base sub(Base d1, Base d2)
	{
		int[] sh=getCommonShape(d1.shape, d2.shape);
		Base res=new Base(sh).setRequiresGradient(d1.requiresGradient() | d2.requiresGradient());
		if (res.requiresGradient())
			res.setGradientFunction(subtractionGradient, d1, d2);
		int len=res.length; // length of the array.
		int[] tmpSh=new int[sh.length]; // temporary shape holder.
		for (int i=0;i < len;i++)
		{
			indexToShape(i, sh, tmpSh);
			float op1=d1.get(tmpSh);
			float op2=d2.get(tmpSh);
			res.setRaw(i, op1 - op2);
		}
		return res;
	}
	public static Base sub(Base d1, float d2)
	{
		int[] sh=d1.shape;
		Base res=new Base(sh).setRequiresGradient(d1.requiresGradient());
		Base data2=new Base(new float[]{d2}); // don't use for computation, it is just for gradient.
		if (res.requiresGradient())
			res.setGradientFunction(subtractionGradient, d1, data2);
		int len=res.length; // length of the array.
		int[] tmpSh=new int[sh.length]; // temporary shape holder.
		for (int i=0;i < len;i++)
		{
			indexToShape(i, sh, tmpSh);
			float op1=d1.get(tmpSh);
			res.setRaw(i, op1 - d2);
		}
		return res;
	}
	// thus functions is optional.
	public static Base sub(float d1, Base d2)
	{
		int[] sh=d2.shape;
		Base res=new Base(sh).setRequiresGradient(d2.requiresGradient());
		Base data1=new Base(new float[]{d1}); // don't use for computation, it is just for gradient.
		if (res.requiresGradient())
			res.setGradientFunction(subtractionGradient, data1, d2);
		int len=res.length; // length of the array.
		int[] tmpSh=new int[sh.length]; // temporary shape holder.
		for (int i=0;i < len;i++)
		{
			indexToShape(i, sh, tmpSh);
			float op2=d2.get(tmpSh);
			res.setRaw(i, d1 - op2);
		}
		return res;
	}
	// multiplication
	public static Base mul(Base d1, Base d2)
	{
		int[] sh=getCommonShape(d1.shape, d2.shape);
		Base res=new Base(sh).setRequiresGradient(d1.requiresGradient() | d2.requiresGradient());
		if (res.requiresGradient())
			res.setGradientFunction(multiplicationGradient, d1, d2);
		int len=res.length; // length of the array.
		int[] tmpSh=new int[sh.length]; // temporary shape holder.
		for (int i=0;i < len;i++)
		{
			indexToShape(i, sh, tmpSh);
			float op1=d1.get(tmpSh);
			float op2=d2.get(tmpSh);
			res.setRaw(i, op1 * op2);
		}
		return res;
	}
	public static Base mul(Base d1, float d2)
	{
		int[] sh=d1.shape;
		Base res=new Base(sh).setRequiresGradient(d1.requiresGradient());
		Base data2=new Base(new float[]{d2}); // don't use for computation, it is just for gradient.
		if (res.requiresGradient())
			res.setGradientFunction(multiplicationGradient, d1, data2);
		int len=res.length; // length of the array.
		int[] tmpSh=new int[sh.length]; // temporary shape holder.
		for (int i=0;i < len;i++)
		{
			indexToShape(i, sh, tmpSh);
			float op1=d1.get(tmpSh);
			res.setRaw(i, op1 * d2);
		}
		return res;
	}
	// division
	public static Base div(Base d1, Base d2)
	{
		int[] sh=getCommonShape(d1.shape, d2.shape);
		Base res=new Base(sh);
		int len=res.length; // length of the array.
		int[] tmpSh=new int[sh.length]; // temporary shape holder.
		for (int i=0;i < len;i++)
		{
			indexToShape(i, sh, tmpSh);
			float op1=d1.get(tmpSh);
			float op2=d2.get(tmpSh);
			res.setRaw(i, op2 == 0 ?0: op1 / op2);
		}
		return res;
	}
	public static Base div(Base d1, float d2)
	{
		int[] sh=d1.shape;
		Base res=new Base(sh);
		if (d2 == 0)
			return res;
		int len=res.length; // length of the array.
		int[] tmpSh=new int[sh.length]; // temporary shape holder.
		for (int i=0;i < len;i++)
		{
			indexToShape(i, sh, tmpSh);
			float op1=d1.get(tmpSh);
			res.setRaw(i, op1 / d2);
		}
		return res;
	}

	// this functions is optional.
	public static Base div(float d1, Base d2)
	{
		int[] sh=d2.shape;
		Base res=new Base(sh);
		int len=res.length; // length of the array.
		int[] tmpSh=new int[sh.length]; // temporary shape holder.
		for (int i=0;i < len;i++)
		{
			indexToShape(i, sh, tmpSh);
			float op2=d2.get(tmpSh);
			res.setRaw(i, op2 == 0 ?0: d1 / op2);
		}
		return res;
	}
	// power function.
	public static Base pow(Base d1, Base d2)
	{
		int[] sh=getCommonShape(d1.shape, d2.shape);
		Base res=new Base(sh).setRequiresGradient(d1.requiresGradient() | d2.requiresGradient());
		if (res.requiresGradient())
			res.setGradientFunction(powGradient, d1, d2);
		int len=res.length; // length of the array.
		int[] tmpSh=new int[sh.length]; // temporary shape holder.
		for (int i=0;i < len;i++)
		{
			indexToShape(i, sh, tmpSh);
			float op1=d1.get(tmpSh);
			float op2=d2.get(tmpSh);
			res.setRaw(i, (float)Math.pow(op1 , op2));
		}
		return res;
	}
	public static Base pow(Base d1, float d2)
	{
		int[] sh=d1.shape;
		Base res=new Base(sh).setRequiresGradient(d1.requiresGradient());
		Base data2=new Base(new float[]{d2}); // don't use for computation, it is just for gradient.
		if (res.requiresGradient())
			res.setGradientFunction(powGradient, d1, data2);

		int len=res.length; // length of the array.
		int[] tmpSh=new int[sh.length]; // temporary shape holder.
		for (int i=0;i < len;i++)
		{
			indexToShape(i, sh, tmpSh);
			float op1=d1.get(tmpSh);
			res.setRaw(i, (float)Math.pow(op1 , d2));
		}
		return res;
	}
	public static Base pow(float d1, Base d2)
	{
		int[] sh=d2.shape;
		Base res=new Base(sh).setRequiresGradient(d2.requiresGradient());
		Base data1=new Base(new float[]{d1}); // don't use for computation, it is just for gradient.
		if (res.requiresGradient())
			res.setGradientFunction(powGradient, data1, d2);
		int len=res.length; // length of the array.
		int[] tmpSh=new int[sh.length]; // temporary shape holder.
		for (int i=0;i < len;i++)
		{
			indexToShape(i, sh, tmpSh);
			float op2=d2.get(tmpSh);
			res.setRaw(i, (float)Math.pow(d1, op2));
		}
		return res;
	}
	// dot product start.

	public static Base dot(Base a, Base b)
	{
		int[] out=dotShape(a.shape, b.shape);
		a = a.as2DArray();
		if (b.getDim() < 2)
			b = b.reshape(1, -1);
		else if (b.getDim() == 2)
		{
			b = b.transpose(1, 0);
		}
		else if (b.getDim() > 2)
		{
			b = b.transpose(dotAxis(b.getDim()));
			b = b.reshape(-1, b.shape[b.shape.length - 1]);
		}
		if (b.shape[b.shape.length - 1] != a.shape[a.shape.length - 1])
			throw new RuntimeException("invalid shape dor dot product.");
		int[] sh={a.shape[0],b.shape[0]};
		Base bs=new Base(sh);
		// float[] outData=new float[a.shape[0] * b.shape[0]];
		for (int ar=0;ar < a.shape[0];ar++)
			for (int br=0;br < b.shape[0];br++)
			{
				float sm=0;	
				for (int c=0;c < a.shape[1];c++)
				{
					sm += a.get(ar, c) * b.get(br, c);
				}
				bs.set(new int[]{ar,br}, sm);
			}
		bs.setRequiresGradient(a.requiresGradient() | b.requiresGradient());
		bs = bs.setGradientFunction(GradFunc.dotGradient, a, b).reshape(out);
		return bs;
	}
	public static int[] dotAxis(int len)
	{
		// the len value expectes to be > 2
		int[]a=new int[len];
		for (int i=0;i < len;i++)
			a[i] = i;
		int t=a[len - 1];
		a[len - 1] = a[len - 2];
		a[len - 2] = t;
		return a;
	}
	public static int[] dotShape(int[]sh1, int[]sh2)
	{
		// this method expect the array before transposed.
		if (n(sh1, 0) != n(sh2, sh2.length == 1 ?0: 1))
			throw new RuntimeException("incompatable shape for dot product.");
		int len=sh1.length + sh2.length - 2;
		// print("length :", len);
		if (len == 0)
			return new int[]{1};
		int[]ns=new int[len];
		for (int i=0;i < sh1.length - 1;i++)
			ns[i] = sh1[i];
		int sh2Start=sh1.length - 1;
		for (int i=0;i < sh2.length - 2;i++)
			ns[i + sh2Start] = sh2[i];
		if (sh2.length >= 2)
			ns[ns.length - 1] = sh2[sh2.length - 1];
		return ns;
	}
	// dot product end.
	public static Base convolve1d(Base d1, Base kr)
	{
		// this convolve only support 1d kernel.
		// kr = kr.trim();
		if (kr.shape.length != 1)
			throw new RuntimeException("convolve 1d error : expected 1d kernel array");
		d1 = d1.as2DArray();
		int len=d1.shape[1] - kr.shape[0] + 1;
		int[]outShape={d1.shape[0],len};
		float[] out=new float[d1.shape[0] * len];
		for (int dr=0;dr < d1.shape[0];dr++)
		{
			// convolve mode normal.
			for (int w=0;w < len;w++)
			{
				float sm=0;
				int kp=kr.shape[0] - 1;
				for (int k=0;k < kr.shape[0];k++)
				{
					float kv=kr.get(kp);
					// cache kv on array for next time to speedup.
					sm += d1.get(dr, w + k) * kv; // get kernel in reverse order
					kp--; // count downward
				}
				out[shapeToIndex(new int[]{dr,w}, outShape)] = sm;
			}
		}
		return new Base(out, outShape);
	}
	public static Base correlate1d(Base d1, Base kr)
	{
		// this convolve, only support 1d kern.
		kr = kr.trim();
		if (kr.trim().shape.length != 1)
			throw new RuntimeException("correlation 1d error : expected 1d kernel array");
		d1 = d1.as2DArray();
		int len=d1.shape[1] - kr.shape[0] + 1;
		int[]outShape={d1.shape[0],len};
		float[] out=new float[d1.shape[0] * len];
		for (int dr=0;dr < d1.shape[0];dr++)
		{
			// iterate over rows of input data.
			// convolve mode normal.
			for (int w=0;w < len;w++)
			{
				float sm=0;
				for (int k=0;k < kr.shape[0];k++)
				{
					float kv=kr.get(k);
					sm += d1.get(dr, w + k) * kv; // get kernel in reverse order
				}
				out[shapeToIndex(new int[]{dr,w}, outShape)] = sm;
			}		
		}
		return new Base(out, outShape);
	}
	public static Base fullCorrelate1d(Base a, Base b)
	{
		// gradient not implemented.
		// only supported 1d array of input and kernel.
		if (a.getDim() != 1 || b.getDim() != 1)
		{
			throw new RuntimeException("invalid array size.");
		}
		int outSize=(a.shape[0] + b.shape[0]) - 1;
		Base out = new Base(outSize);
		for (int i=0;i < outSize;i++) // increment by inc. default = 1.
		{
			float sm=0;
			int ips=(i - b.shape[0]) + 1;
			for (int k=0;k < b.shape[0];k++)
			{
				float kv=b.get(k); //  kernels.get(kr, chn, kp);
				int kp=ips + k;
				if (kp >= 0 && kp < a.shape[0])
					sm += a.get(kp) * kv;
			}
			out.set(ar(i), sm);
		}
		return out;
	}
}
