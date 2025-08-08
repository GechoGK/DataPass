package gss.arr;

import gss.*;
import java.util.*;

import static gss.Util.*;
import static gss.arr.GradFunc.*;

public class NDArray
{
	/*
	 -- dot product needs more improvement.
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
			res.setRaw(i, op1 + op2);
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
			res.setRaw(i, op1 + d2);
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
	public static Base dot(Base d1, Base d2)
	{
		// fix transposing and reshaping b array
		// only transposing in enough.
		/*
		 if (x.col != y.row)
		 throw new RuntimeException("two matrixes doesn't match (" + x.col + "," + y.row + ")");
		 VMat m=new VMat(x.row, y.col);
		 for (int r=0;r < x.row;r++)
		 for (int c=0;c < y.col;c++)
		 {
		 Value sum=newValue(0);
		 for (int i=0;i < x.col;i++)
		 sum = Value.add(sum, Value.mul(x.get(r, i) , y.get(i, c)));
		 m.put(r, c, sum);
		 }
		 return m;
		 */
		if (d1.shape.length == 1)
			d1 = d1.as2DArray();
		if (d2.getDim() == 1)
			d2 = d2.as2DArray();
		int[] newShape=getShapeForDot(d1.shape, d2.shape);
		// System.out.println("final shape :" + Arrays.toString(newShape));
		d1 = d1.as2DArray();

		d2 = d2.transpose(prepareAxisForDot(d2.shape.length)).reshape(d2.shape[d2.shape.length - 2], -1);
		int[]sh1=d1.shape;
		int[]sh2=d2.shape;
		if (sh1[1] != sh2[0])
			throw new RuntimeException("two matrixes doesn't match (" + sh1[1] + "," + sh2[0] + ")");
		int[] dotShape={sh1[0],sh2.length == 1 ?1: sh2[1]};
		// System.out.println(".." + Arrays.toString(sh1) + ".." + Arrays.toString(sh2) + "..." + Arrays.toString(dotShape));
		float[] f=new float[dotShape[0] * dotShape[1]];

		for (int r=0;r < sh1[0];r++)
			for (int c=0;c < dotShape[1];c++)
			{
				float sum=0;
				for (int i=0;i < sh1[1];i++)
					sum += d1.get(r, i) * d2.get(i, c);
				f[shapeToIndex(ar(r, c), dotShape)] = sum;
			}

		Base d = new Base(f, dotShape).setRequiresGradient(d1.requiresGradient() | d2.requiresGradient());
		if (d.requiresGradient())
			d.setGradientFunction(GradFunc.dotGradient, d1, d2);
		return d.reshape(newShape);
	}
	public static int[] prepareAxisForDot(int len)
	{
		/*
		 how to transpose the 2nd array inorder to get expected shape.
		 // 0,1,2     -> 2,1,0     -> 1,0,2
		 // 3,2,1,0   -> 3,2,0,1   -> 2,0,1,3
		 // 4,3,2,1,0 -> 4,3,0,1,2 -> 3,0,1,2,4
		 // reverse   -> swap      -> shift
		 // there is always shift.
		 // number of swaps = dim - 2 from the left before swap.
		 */
		int[] ax=new int[len];
		// how to swap axis.
		// reverse, flip(swap) , shift
		// ax[ax.length - 1] = tmp;
		// reverse;
		for (int i=0;i < ax.length;i++)
			ax[i] = ax.length - i - 1;
		// System.out.println("reversed axis" + Arrays.toString(ax));
		// swap
		int sLen=ax.length - 2;
		for (int i=0;i < sLen / 2;i++)
		{
			int cp=ax.length - sLen + i;
			int lp=ax.length - i - 1;
			int cv=ax[cp];
			int lv=ax[lp];
			ax[cp] = lv;
			ax[lp] = cv;
			// System.out.println("[" + cp + " <-> " + lp + "], (" + cv + ", " + lv + ")");
		}
		// System.out.println("swapped axis " + Arrays.toString(ax));
		// left shift
		int tmp=ax[0];
		for (int i=0;i < ax.length - 1;i++)
			ax[i] = ax[i + 1];
		ax[ax.length - 1] = tmp;
		// System.out.println("shifted axis :" + Arrays.toString(ax));
		return ax;
	}
	private static int[] getShapeForDot(int[]s1, int[]s2)
	{
		/*
		 output shape is determined by the given array's shape.
		 example  a.shape = (x,y,z)
		 ..       b.shape = (h,i,j) then
		 ..
		 ..       c = a.dot(b)
		 ..
		 .. first we need to check if "z" and "i" are equal if true.
		 the output shape(c.shape) would be (x,y,h,j) it increase by 1 dimension.

		 */
		if (n(s1, 0) != n(s2, 1))
			throw new RuntimeException("dimensions are not equal to compute the dot product (" + n(s1, 0) + " != " + n(s2, 1) + ")");
		int[] newShape=new int[s1.length + s2.length - 2];

		for (int i=0;i < s1.length - 1;i++)
			newShape[i] = s1[i];
		int str=s1.length - 1;
		for (int i=0;i < s2.length - 1;i++)
			newShape[str + i] = s2[i];
		newShape[newShape.length - 1] = s2[s2.length - 1];
		// System.out.println("new Shape =" + Arrays.toString(newShape));
		return newShape;
	}
	// dot product end.
	public static Base convolve1d(Base d1, Base kr)
	{
		// this convolve only support 1d, 1d kern.
		kr = kr.trim();
		if (kr.trim().shape.length != 1)
			throw new RuntimeException("convolve 1d error : expected 1d kernel array");
		d1 = d1.as2DArray();
		int len=d1.shape[1] - kr.shape[0] + 1;
		int[]outShape={d1.shape[0],len};
		float[] out=new float[d1.shape[0] * len];
		for (int dr=0;dr < d1.shape[0];dr++)
		{
			// iterate over rows of input data.
			if (dr == 0)
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
