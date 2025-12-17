package gss.arr;

import java.util.*;

import static gss.Util.*;
import static gss.arr.GradFunc.*;
import static gss.Functions.*;

public class NDArray
{
	/*
	 -- remove all setRaw functions.
	 */
	// array generators.
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
	public static Base empty(int...shape)
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
	public static Base wrap(float...v)
	{
		return new Base(v, new int[]{v.length});
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
	public static Base zip(Base p1, Base p2, ZipFunction zipFunc)
	{
		int[] sh=getCommonShape(p1.shape, p2.shape);
		Base res=new Base(sh).setRequiresGradient(p1.hasGradient() || p2.hasGradient());
		int len=res.length; // length of the array.
		int[] tmpSh=new int[sh.length]; // temporary shape holder.
		for (int i=0;i < len;i++)
		{
			indexToShape(i, sh, tmpSh);
			float op1=p1.get(tmpSh);
			float op2=p2.get(tmpSh);
			res.set(tmpSh, zipFunc.apply(op1, op2));
		}
		return res;
	}
	// addition
	public static Base add(Base d1, Base d2)
	{
		Base out=zip(d1, d2, new ZipFunction(){
				@Override
				public float apply(float p1, float p2)
				{
					return p1 + p2;
				}
			});
		return out.setGradientFunctionS(additionGradient, d1, d2);
	}
	public static Base add(Base p1, float p2)
	{
		int[] sh=p1.shape;
		Base res=new Base(sh).setRequiresGradient(p1.hasGradient());
		Base data2=new Base(new float[]{p2}); // don't use for computation, it is just for gradient.
		if (res.hasGradient())
			res.setGradientFunction(additionGradient, p1, data2);
		int len=res.length; // length of the array.
		int[] tmpSh=new int[sh.length]; // temporary shape holder.
		for (int i=0;i < len;i++)
		{
			indexToShape(i, sh, tmpSh);
			float op1=p1.get(tmpSh);
			res.set(tmpSh, op1 + p2);
		}
		return res;
	}
	// subtraction
	public static Base sub(Base p1, Base p2)
	{
		int[] sh=getCommonShape(p1.shape, p2.shape);
		Base res=new Base(sh).setRequiresGradient(p1.hasGradient() | p2.hasGradient());
		if (res.hasGradient())
			res.setGradientFunction(subtractionGradient, p1, p2);
		int len=res.length; // length of the array.
		int[] tmpSh=new int[sh.length]; // temporary shape holder.
		for (int i=0;i < len;i++)
		{
			indexToShape(i, sh, tmpSh);
			float op1=p1.get(tmpSh);
			float op2=p2.get(tmpSh);
			res.setRaw(i, op1 - op2);
		}
		return res;
	}
	public static Base sub(Base p1, float p2)
	{
		int[] sh=p1.shape;
		Base res=new Base(sh).setRequiresGradient(p1.hasGradient());
		Base data2=new Base(new float[]{p2}); // don't use for computation, it is just for gradient.
		if (res.hasGradient())
			res.setGradientFunction(subtractionGradient, p1, data2);
		int len=res.length; // length of the array.
		int[] tmpSh=new int[sh.length]; // temporary shape holder.
		for (int i=0;i < len;i++)
		{
			indexToShape(i, sh, tmpSh);
			float op1=p1.get(tmpSh);
			res.setRaw(i, op1 - p2);
		}
		return res;
	}
	// thus functions is optional.
	public static Base sub(float p1, Base p2)
	{
		int[] sh=p2.shape;
		Base res=new Base(sh).setRequiresGradient(p2.hasGradient());
		Base data1=new Base(new float[]{p1}); // don't use for computation, it is just for gradient.
		if (res.hasGradient())
			res.setGradientFunction(subtractionGradient, data1, p2);
		int len=res.length; // length of the array.
		int[] tmpSh=new int[sh.length]; // temporary shape holder.
		for (int i=0;i < len;i++)
		{
			indexToShape(i, sh, tmpSh);
			float op2=p2.get(tmpSh);
			res.setRaw(i, p1 - op2);
		}
		return res;
	}
	// multiplication
	public static Base mul(Base p1, Base p2)
	{
		int[] sh=getCommonShape(p1.shape, p2.shape);
		Base res=new Base(sh).setRequiresGradient(p1.hasGradient() | p2.hasGradient());
		if (res.hasGradient())
			res.setGradientFunction(multiplicationGradient, p1, p2);
		int len=res.length; // length of the array.
		int[] tmpSh=new int[sh.length]; // temporary shape holder.
		for (int i=0;i < len;i++)
		{
			indexToShape(i, sh, tmpSh);
			float op1=p1.get(tmpSh);
			float op2=p2.get(tmpSh);
			res.setRaw(i, op1 * op2);
		}
		return res;
	}
	public static Base mul(Base p1, float p2)
	{
		int[] sh=p1.shape;
		Base res=new Base(sh).setRequiresGradient(p1.hasGradient());
		Base data2=new Base(new float[]{p2}); // don't use for computation, it is just for gradient.
		if (res.hasGradient())
			res.setGradientFunction(multiplicationGradient, p1, data2);
		int len=res.length; // length of the array.
		int[] tmpSh=new int[sh.length]; // temporary shape holder.
		for (int i=0;i < len;i++)
		{
			indexToShape(i, sh, tmpSh);
			float op1=p1.get(tmpSh);
			res.setRaw(i, op1 * p2);
		}
		return res;
	}
	// division
	public static Base div(Base p1, Base p2)
	{
		int[] sh=getCommonShape(p1.shape, p2.shape);
		Base res=new Base(sh);
		int len=res.length; // length of the array.
		int[] tmpSh=new int[sh.length]; // temporary shape holder.
		for (int i=0;i < len;i++)
		{
			indexToShape(i, sh, tmpSh);
			float op1=p1.get(tmpSh);
			float op2=p2.get(tmpSh);
			res.setRaw(i, op2 == 0 ?0: op1 / op2);
		}
		return res;
	}
	public static Base div(Base p1, float p2)
	{
		int[] sh=p1.shape;
		Base res=new Base(sh);
		if (p2 == 0)
			return res;
		int len=res.length; // length of the array.
		int[] tmpSh=new int[sh.length]; // temporary shape holder.
		for (int i=0;i < len;i++)
		{
			indexToShape(i, sh, tmpSh);
			float op1=p1.get(tmpSh);
			res.setRaw(i, op1 / p2);
		}
		return res;
	}

	// this functions is optional.
	public static Base div(float p1, Base p2)
	{
		int[] sh=p2.shape;
		Base res=new Base(sh);
		int len=res.length; // length of the array.
		int[] tmpSh=new int[sh.length]; // temporary shape holder.
		for (int i=0;i < len;i++)
		{
			indexToShape(i, sh, tmpSh);
			float op2=p2.get(tmpSh);
			res.setRaw(i, op2 == 0 ?0: p1 / op2);
		}
		return res;
	}
	// power function.
	public static Base pow(Base p1, Base p2)
	{
		int[] sh=getCommonShape(p1.shape, p2.shape);
		Base res=new Base(sh).setRequiresGradient(p1.hasGradient() | p2.hasGradient());
		if (res.hasGradient())
			res.setGradientFunction(powGradient, p1, p2);
		int len=res.length; // length of the array.
		int[] tmpSh=new int[sh.length]; // temporary shape holder.
		for (int i=0;i < len;i++)
		{
			indexToShape(i, sh, tmpSh);
			float op1=p1.get(tmpSh);
			float op2=p2.get(tmpSh);
			res.setRaw(i, (float)Math.pow(op1 , op2));
		}
		return res;
	}
	public static Base pow(Base p1, float p2)
	{
		int[] sh=p1.shape;
		Base res=new Base(sh).setRequiresGradient(p1.hasGradient());
		Base data2=new Base(new float[]{p2}); // don't use for computation, it is just for gradient.
		if (res.hasGradient())
			res.setGradientFunction(powGradient, p1, data2);

		int len=res.length; // length of the array.
		int[] tmpSh=new int[sh.length]; // temporary shape holder.
		for (int i=0;i < len;i++)
		{
			indexToShape(i, sh, tmpSh);
			float op1=p1.get(tmpSh);
			res.setRaw(i, (float)Math.pow(op1 , p2));
		}
		return res;
	}
	public static Base pow(float p1, Base p2)
	{
		int[] sh=p2.shape;
		Base res=new Base(sh).setRequiresGradient(p2.hasGradient());
		Base data1=new Base(new float[]{p1}); // don't use for computation, it is just for gradient.
		if (res.hasGradient())
			res.setGradientFunction(powGradient, data1, p2);
		int len=res.length; // length of the array.
		int[] tmpSh=new int[sh.length]; // temporary shape holder.
		for (int i=0;i < len;i++)
		{
			indexToShape(i, sh, tmpSh);
			float op2=p2.get(tmpSh);
			res.setRaw(i, (float)Math.pow(p1, op2));
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
		bs.setRequiresGradient(a.hasGradient() | b.hasGradient());
		if (bs.hasGradient())
			bs.setGradientFunction(GradFunc.dotGradient, a, b);
		return bs.reshape(out);
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
	public static Base convolve1d(Base p1, Base kr)
	{
		return convolve1d(p1, kr, 0); // by default mode = normal.
	}
	public static Base convolve1d(Base p1, Base kr, int mode)
	{
		Base out=null;
		if (mode == 0) // mode valid
			out = convolve1dValid(p1, kr, null);
		else if (mode == 1) // mode full.
			out = convolve1dFull(p1, kr, null);
		if (out == null)
			return null;
		out.setRequiresGradient(p1.hasGradient() | kr.hasGradient());
		if (out.hasGradient())
			out.setGradientFunction(GradFunc.convolveGradient, p1, kr);
		return out;
	}
	public static Base convolve1dValid(Base a, Base b, Base out)
	{
		/*
		 example 1.
		 a = [1,2,3] any data.
		 b = [2,3,4] mostly kernel. and flipped.
		 c = a @ b
		 = 1*4 + 2*3 + 3*2 = 4 + 6 + 6 = 16.
		 example 2
		 a = [1,2,3,4,5] any data.
		 b = [2,3,4] mostly kernel. and flipped.
		 c = a @ b
		 = 1*4 + 2*3 + 3*2 = 4 + 6 + 6    = 16.
		 = 2*4 + 3*3 + 4*2 = 8 + 9 + 8    = 25
		 = 3*4 + 4*3 + 5*2 = 12 + 12 + 10 = 34

		 */
		// this convolve only support 1d data and 1d kernel.
		if (a.getDim() != 1 || b.getDim() != 1)
			throw new RuntimeException("convolve 1d error : expected 1d kernel array and 1d data.");
		int len=Math.max(a.length, b.length) - Math.min(a.length, b.length) + 1;
		if (out == null)
			out = new Base(len);
		int wi=0;
		int iinc=a.length > b.length ?1: 0;
		int kinc=b.length > a.length ?1: 0;
		int wr=b.length > a.length ?len - 1: 0;
		for (int w=0;w < len;w++)
		{
			float sm=0;
			int kl=b.length - 1;
			for (int i=0;i < Math.min(a.length, b.length);i++)
			{
				float aa=b.get(kl - wr);
				float bb=a.get(i + wi);
				sm += aa * bb;
				kl--;
			}
			wr -= kinc;
			wi += iinc;
			out.set(ar(w), sm);
		}
		return out;
	}
	public static Base convolve1dFull(Base a, Base b, Base outData)
	{
		/*
		 example 1.
		 a = [1,2,3] any data.
		 b = [2,3,4] mostly kernel. and flipped.
		 c = a @f b
		 = [0,0,1]
		 = [4,3,2] = 1*2 = 2
		 = [0,1,2]
		 = [4,3.2] = 2*2 + 1*3 = 4 + 3 = 7
		 = [1,2,3]
		 = [4,3,2] = 1*4 + 2*3 + 3*2 = 4 + 6 + 6 = 16.
		 = [2,3,0]
		 = [4,3,2] = 2*4 + 3*3 = 8 + 9 = 17
		 = [3,0,0]
		 = [4,3,2] = 3*4 = 12
		 example 2 // not tested.
		 a = [1,2,3,4,5] any data.
		 b = [2,3,4] mostly kernel. and flipped.
		 c = a @f b
		 = [0,0,1]
		 = [4,3,2] = 1*2 = 2
		 = [0,1,2]
		 = [4,3.2] = 2*2 + 1*3 = 4 + 3 = 7
		 = [1,2,3]
		 = [4,3,2] = 1*4 + 2*3 + 3*2 = 4 + 6 + 6 = 16.
		 = [2,3,4]
		 = [4,3,2] = 2*4 + 3*3 + 4*2 = 8+9+8=25
		 = [3,4,5]
		 = [4,3,2] = 3*4 + 4*3 + 5*2 = 12+12+10 = 34
		 = [4,5,0]
		 = [4,3,2] = 4*4 + 5*3 = 16+15 =31
		 = [5,0,0]
		 = [4,3,2] = 5*4 = 20
		 */
		// only supported 1d array of input and kernel.
		if (a.getDim() != 1 || b.getDim() != 1)
			throw new RuntimeException("convolution 1d error : expected 1d kernel array and 1d data");;
		int len=(a.shape[0] + b.shape[0]) - 1;
		if (outData == null)
			outData = new Base(len);
		for (int i=0;i < len;i++) // increment by inc. default = 1.
		{
			float sm=0;
			int kr=b.length - 1;
			int ips=-(b.length - 1) + i;
			for (int k=0;k < b.length;k++)
			{
				int kp=ips + k;
				if (kp >= 0 && kp < a.shape[0])
				{
					float kv=b.get(kr); //  kernels.get(kr, chn, kp);
					sm += a.get(kp) * kv;
				}
				kr--;
			}
			outData.set(ar(i) , sm);
		}
		return outData;
	}
	public static Base correlate1d(Base p1, Base kr)
	{
		return correlate1d(p1, kr, 0);
	}
	public static Base correlate1d(Base p1, Base kr, int mode)
	{
		Base out=null;
		if (mode == 0) // mode valid.
			out = correlate1dValid(p1, kr, null);
		else if (mode == 1) // modr full.
			out = correlate1dFull(p1, kr, null);
		return out;
	}
	public static Base correlate1dValid(Base a, Base b, Base f)
	{
		if (a.getDim() != 1 || b.getDim() != 1)
			throw new RuntimeException("correlation 1d error : expected 1d kernel array and 1d data");
		int len=Math.max(a.length, b.length) - Math.min(a.length, b.length) + 1;
		if (f == null)
			f = new Base(len);
		int wi=0;
		int iinc=a.length > b.length ?1: 0;
		int kinc=b.length > a.length ?1: 0;
		int wr=b.length > a.length ?len - 1: 0;
		for (int w=0;w < len;w++)
		{
			float sm=0;
			int kl=0;
			for (int i=0;i < Math.min(a.length, b.length);i++)
			{
				float aa=b.get(i + wr);
				float bb=a.get(i + wi);
				sm += aa * bb;
				kl++;
			}
			wr -= kinc;
			wi += iinc;
			f.set(ar(w), sm);
		}
		return f;
	}
	public static Base correlate1dFull(Base a, Base b, Base outData)
	{
		// gradient not implemented.
		// only supported 1d array of input and kernel.
		if (a.getDim() != 1 || b.getDim() != 1)
			throw new RuntimeException("correlation 1d error : expected 1d kernel array and 1d data");;
		int len=(a.shape[0] + b.shape[0]) - 1;
		if (outData == null)
			outData = new Base(len);
		for (int i=0;i < len;i++) // increment by inc. default = 1.
		{
			float sm=0;
			int ips=-(b.length - 1) + i;
			for (int k=0;k < b.length;k++)
			{
				int kp=ips + k;
				if (kp >= 0 && kp < a.shape[0])
				{
					float kv=b.get(k); //  kernels.get(kr, chn, kp);
					sm += a.get(kp) * kv;
				}
			}
			outData.set(ar(i) , sm);
		}
		return outData;
	}
	public static Base sum(Base b, int axis)
	{
		error("the sum function is not available at the moment, please review the code.");
		return null;
	}
	// new functions.

}
