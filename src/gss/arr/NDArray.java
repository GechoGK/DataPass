package gss.arr;

import gss.*;
import java.io.*;
import java.util.*;
import org.json.*;

import static gss.Util.*;
import static gss.arr.GradFunc.*;
import static gss.Functions.*;

public class NDArray
{
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
	public static Base arange(float lim, int[]shape)
	{
		float len=length(shape);
		float inc= lim / len;
		return arange(0, lim, inc).reshapeLocal(shape);
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
		float[][]f=new float[size][size];
		for (int i=0;i < size;i++)
			f[size][size] = 1;
		Base d=new Base(flatten(f), size, size);
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
	public static Base wrap(float[][]data)
	{
		int[] shp={data.length,data[0].length};
		float[] dt=Util.flatten(data); // avoid it.
		return new Base(dt, shp);
	}
	public static Base wrap(float[][][]data)
	{
		int[] shp={data.length,data[0].length,data[0][0].length};
		float[] dt=Util.flatten(data); // avoid it.
		return new Base(dt, shp);
	}
	public static Base wrap(float[][][][]data)
	{
		int[] shp={data.length,data[0].length,data[0][0].length,data[0][0][0].length};
		float[] dt=Util.flatten(data); // avoid it.
		return new Base(dt, shp);
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
		Base res=empty(sh).setRequiresGradient(p1.hasGradient() || p2.hasGradient());
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
	public static Base map(Base b, MapFunction mapFunc)
	{
		Base res=empty(b.shape).setRequiresGradient(b.hasGradient());
		int len=res.length;
		for (int i=0;i < len;i++)
		{
			float v=b.get1d(i);
			res.set1d(i, mapFunc.apply(v));
		}
		return res;
	}
	public static Base map(int[]shape, ArrayFunction func)
	{
		float[]data=loop(shape, func);
		Base b=NDArray.wrap(data, shape);
		return b;
	}
	// @not_used.
	public static Base assign(Base b, MapFunction mapFunc)
	{
		int len=b.length;
		for (int i=0;i < len;i++)
		{
			float v=b.get1d(i);
			b.set1d(i, mapFunc.apply(v));
		}
		return b;
	}
	public static Base reduce(Base b, int[]axis, boolean keepDim, ArrayFunction func)
	{
		int[]sh=copy(b.shape);
		float[] data=loop(sh, axis, func);
		int[]newShape=fromAxis(sh, axis);
		if (!keepDim)
			newShape = remove(newShape, axis);
		Base out=NDArray.wrap(data, newShape).setRequiresGradient(b.hasGradient());
		return out;
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
	public static Base add(Base d1, final float sd2)
	{
		Base out=map(d1, new MapFunction(){
				@Override
				public float apply(float p1)
				{
					return p1 + sd2;
				}
			});
		return out.setGradientFunctionS(additionGradient, d1, wrap(new float[]{sd2}));
	}
	// subtraction
	public static Base sub(Base d1, Base d2)
	{
		Base out=zip(d1, d2, new ZipFunction(){
				@Override
				public float apply(float p1, float p2)
				{
					return p1 - p2;
				}
			});
		return out.setGradientFunctionS(subtractionGradient, d1, d2);
	}
	public static Base sub(Base d1, final float sd2)
	{
		Base out=map(d1, new MapFunction(){
				@Override
				public float apply(float p1)
				{
					return p1 - sd2;
				}
			});
		return out.setGradientFunctionS(subtractionGradient, d1, wrap(new float[]{sd2}));
	}
	public static Base sub(final float sd1, Base d2)
	{
		Base out=map(d2, new MapFunction(){
				@Override
				public float apply(float p1)
				{
					return sd1 - p1;
				}
			});
		return out.setGradientFunctionS(subtractionGradient, wrap(new float[]{sd1}), d2);
	}
	// multiplication
	public static Base mul(Base d1, Base d2)
	{
		Base out=zip(d1, d2, new ZipFunction(){
				@Override
				public float apply(float p1, float p2)
				{
					return p1 * p2;
				}
			});
		return out.setGradientFunctionS(multiplicationGradient, d1, d2);
	}
	public static Base mul(Base d1, final float sd2)
	{
		Base out=map(d1, new MapFunction(){
				@Override
				public float apply(float p1)
				{
					return p1 * sd2;
				}
			});
		return out.setGradientFunctionS(multiplicationGradient, d1, wrap(new float[]{sd2}));
	}
	// division
	public static Base div(Base d1, Base d2)
	{
		Base out=zip(d1, d2, new ZipFunction(){
				@Override
				public float apply(float p1, float p2)
				{
					float epsilon = 1e-7f;
					return p1 / Math.max(epsilon, p2);
				}
			});
		return out.setGradientFunctionS(divisionGradient, d1, d2);
	}
	public static Base div(Base d1, final float sd2)
	{
		Base out=map(d1, new MapFunction(){
				@Override
				public float apply(float p1)
				{
					float epsilon = 1e-7f;
					return p1 / Math.max(epsilon, sd2);
				}
			});
		return out.setGradientFunctionS(divisionGradient, d1, wrap(new float[]{sd2}));
	}
	public static Base div(final float sd1, Base d2)
	{
		Base out=map(d2, new MapFunction(){
				@Override
				public float apply(float p1)
				{
					float epsilon = 1e-7f;
					return sd1 / Math.max(epsilon, p1);
				}
			});
		return out.setGradientFunctionS(divisionGradient, wrap(new float[]{sd1}), d2);
	}
	// power function.
	public static Base pow(Base d1, Base d2)
	{
		Base out=zip(d1, d2, new ZipFunction(){
				@Override
				public float apply(float p1, float p2)
				{
					return (float)Math.pow(p1, p2);
				}
			});
		return out.setGradientFunctionS(powGradient, d1, d2);
	}
	public static Base pow(Base d1, final float sd2)
	{
		Base out=map(d1, new MapFunction(){
				@Override
				public float apply(float p1)
				{
					return (float)Math.pow(p1, sd2);
				}
			});
		return out.setGradientFunctionS(powGradient, d1, wrap(new float[]{sd2}));
	}
	public static Base pow(final float sd1, Base d2)
	{
		Base out=map(d2, new MapFunction(){
				@Override
				public float apply(float p1)
				{
					return (float)Math.pow(sd1, p1);
				}
			});
		return out.setGradientFunctionS(powGradient, wrap(new float[]{sd1}), d2);
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
		print("==",sh);
		// make it faster by caching it.
		// float[] outData=new float[a.shape[0] * b.shape[0]];
		float[][]aa=MathUtil.copy2(a);
		float[][]bb=MathUtil.copy2(b);
		float[][]o=new float[a.shape[0]][b.shape[0]];
		int p=0;
		for (int ar=0;ar < a.shape[0];ar++)
			for (int br=0;br < b.shape[0];br++)
			{
				float sm=0;	
				for (int c=0;c < a.shape[1];c++)
				{
					sm += aa[ar][c] * bb[br][c];
					// sm += a.get(ar, c) * b.get(br, c);
				}
				o[ar][br] = sm;
				// bs.set(new int[]{ar,br}, sm);
			}
		Base bs=NDArray.wrap(o);
		bs.setRequiresGradient(a.hasGradient() | b.hasGradient());
		bs.setGradientFunctionS(GradFunc.dotGradient, a, b);
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
		else if (mode == 1) // mode full.
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
	public static Base sum(final Base b, final int...axis)
	{
		final int[]sh=b.shape;
		final int[]subSh=collect(sh, axis);
		final int length=length(subSh);
		Base out=reduce(b, axis, true, new Functions.ArrayFunction(){
				@Override
				public float apply(int[] p1)
				{
					float sum=0;
					for (int i=0;i < length;i++)
					{
						int[]newSH=indexToShape(i, subSh);
						replace(p1, axis, newSH);
						sum += b.get(p1);
					}
					return sum;
				}
			});
		out.setGradientFunctionS(sumGradient, axis, b);
		return out;
	}
	public static Base sum(Base b)
	{
		float sum=0;
		for (int i=0;i < b.length;i++)
		{
			sum += b.get1d(i);
		}
		Base out=wrap(sum, 1).setRequiresGradient(b.hasGradient());
		out.setGradientFunctionS(sumGradient, b);
		return out;
	}
	public static Base log(Base d)
	{
		// log(0) = Infinity. avoid it. add 0.000001...
		Base out=map(d, new MapFunction(){
				@Override
				public float apply(float p1)
				{
					float epsilon = 1e-7f;
					return (float)Math.log(Math.max(epsilon, p1));
				}
			});
		return out.setGradientFunctionS(logGradient, d);
	}
	public static Base exp(Base d)
	{
		Base out=map(d, new MapFunction(){
				@Override
				public float apply(float p1)
				{
					return (float)Math.exp(p1);
				}
			});
		return out.setGradientFunctionS(expGradient, d);
	}
	public static Base sqrt(Base d)
	{
		Base out=map(d, new MapFunction(){
				@Override
				public float apply(float p1)
				{
					return (float)Math.sqrt(p1);
				}
			});
		return out.setGradientFunctionS(sqrtGradient, d);
	}
	public static Base neg(Base d)
	{
		return mul(d, -1);
	}
	public static Base inv(Base d)
	{
		return div(1, d);
	}
	public static Base mean(Base d)
	{
		return NDArray.div(sum(d), d.length);
	}
	public static Base mean(Base d, int...axis)
	{
		return NDArray.div(sum(d, axis), length(collect(d.shape, axis)));// d.shape[axis]);
	}
	public static Base variance(Base d, int...axis)
	{
		Base m=mean(d, axis);
		Base v0 = NDArray.pow(NDArray.sub(d, m), 2);
		v0 = NDArray.div(NDArray.sum(v0, axis), length(collect(v0.shape, axis)));
		return v0;
	}
	// less than
	public static Base lt(Base d1, Base d2)
	{
		Base out=zip(d1, d2, new ZipFunction(){
				@Override
				public float apply(float p1, float p2)
				{
					return p1 < p2 ?1: 0;
				}
			});
		return out.setGradientFunctionS(pass2Gradient, d1, d2);
	}
	public static Base lt(Base d1, final float sd2)
	{
		Base out=map(d1, new MapFunction(){
				@Override
				public float apply(float p1)
				{
					return p1 < sd2 ?1: 0;
				}
			});
		return out.setGradientFunctionS(pass2Gradient, d1, wrap(new float[]{sd2}));
	}
	public static Base lt(final float sd1, Base d2)
	{
		Base out=map(d2, new MapFunction(){
				@Override
				public float apply(float p1)
				{
					return sd1 < p1 ?1: 0;
				}
			});
		return out.setGradientFunctionS(pass2Gradient, wrap(new float[]{sd1}), d2);
	}
	// equals
	public static Base eq(Base d1, Base d2)
	{
		Base out=zip(d1, d2, new ZipFunction(){
				@Override
				public float apply(float p1, float p2)
				{
					return p1 == p2 ?1: 0;
				}
			});
		return out.setGradientFunctionS(pass2Gradient, d1, d2);
	}
	public static Base eq(Base d1, final float sd2)
	{
		Base out=map(d1, new MapFunction(){
				@Override
				public float apply(float p1)
				{
					return p1 == sd2 ?1: 0;
				}
			});
		return out.setGradientFunctionS(pass2Gradient, d1, wrap(new float[]{sd2}));
	}
	public static Base eq(final float sd1, Base d2)
	{
		Base out=map(d2, new MapFunction(){
				@Override
				public float apply(float p1)
				{
					return sd1 == p1 ?1: 0;
				}
			});
		return out.setGradientFunctionS(pass2Gradient, wrap(new float[]{sd1}), d2);
	}
	public static Base min(Base b)
	{
		float val=Float.MAX_VALUE;
		int index=0;
		for (int i=0;i < b.length;i++)
		{
			float v=b.get1d(i);
			if (v < val)
			{
				val = v;
				index = i;
			}
		}
		Base res=empty(1).setRequiresGradient(b.hasGradient());
		res.set(ar(0), val);
		res.setGradientFunctionS(indexGradient, index, b);
		return res;
	}
	public static Base max(Base b)
	{
		float val=Float.MIN_VALUE;
		int index=0;
		for (int i=0;i < b.length;i++)
		{
			float v=b.get1d(i);
			if (v > val)
			{
				val = v;
				index = i;
			}
		}
		Base res=empty(1).setRequiresGradient(b.hasGradient());
		res.set(ar(0), val);
		res.setGradientFunctionS(indexGradient, index, b);
		return res;
	}
	// axis is determined by the shorter shape.length.
	public static Base concat(final Base b1, final Base b2, final int ax)
	{
		int diff1=Math.abs(b1.shape.length - b2.shape.length);
		final int axis=ax + diff1;
		int[]outShape1=getCommonShapeExcept(b1.shape, b2.shape, axis);

		final int[] i1_shape=copy(outShape1);
		final int[] i2_shape=copy(outShape1);
		copyB(b1.shape, i1_shape);
		copyB(b2.shape, i2_shape);

		final int diff2=i1_shape[axis];

		int[] outShape=concatShape(i1_shape, i2_shape, axis);
		Base b=map(outShape, new ArrayFunction(){
				@Override
				public float apply(int[] p1)
				{
					if (p1[axis] >= i1_shape[axis])
					{
						p1[axis] = p1[axis] - diff2;
						return b2.get(p1);
					}
					return b1.get(p1);
				}
			});
		b.setRequiresGradient(b1.hasGradient() || b2.hasGradient());
		b.setGradientFunctionS(concatGradient, ax, b1, b2);
		return b;
	}
	public static Base onehot(Base ind, int vocab_size)
	{
		float[][]out=new float[ind.length][];
		for (int i=0;i < out.length;i++)
		{
			float[] f=new float[vocab_size];
			f[(int)ind.get1d(i)] = 1;
			out[i] = f;
		}
		Base b=NDArray.wrap(out).reshapeLocal(append(ind.shape, vocab_size));
		b.setRequiresGradient(ind.hasGradient());
		b.setGradientFunctionS(stepGradient, ind);
		return b;
	}
	public static Base convolv2d(Base in, Base kern)
	{
		float[][] out=MathUtil.conv2d(in, kern);
		Base o=NDArray.wrap(out);
		o.setRequiresGradient(in.hasGradient() || kern.hasGradient());
		// o.setGradientFunctionS(convolve2DGradient);
		return o;
	}
}
