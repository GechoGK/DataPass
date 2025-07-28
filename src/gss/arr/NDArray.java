package gss.arr;

import gss.*;
import java.util.*;

import static gss.Util.*;

public class NDArray
{
	public static Data arange(float end)
	{
		return arange(0, end, 1);
	}
	public static Data arange(float str, float end)
	{
		return arange(str, end, 1);
	}
	public static Data arange(float str, float end, float inc)
	{
		float[] f=range(str, end, inc);
		Data ar=new Data(f);
		return ar;
	}
	public static Data ones(int...shape)
	{
		return wrap(1, shape);
	}
	public static Data zeros(int...shape)
	{
		return wrap(0, shape);
	}
	public static Data idt(int size)
	{
		int[] sh={size,size};
		float[]f=new float[length(sh)];
		for (int i=0;i < size;i++)
			f[shapeToIndex(ar(size, size), sh)] = 1;
		Data d=new Data(f, sh);
		return d;
	}
	public static Data wrap(float[]v, int...sh)
	{
		return new Data(v, sh);
	}
	public static Data wrap(float v, int...shape)
	{
		int len=length(shape);
		float[] f=new float[len];
		Arrays.fill(f, v);
		return new Data(f, shape);
	}
	public static Data rand(int...shape)
	{
		return rand(shape, -1); // change -1 into another to use seed value.
	}
	public static Data rand(int[]shape, int seed)
	{
		Data arr=new Data(shape);
		Random r=null;
		if (seed != -1)
			r = new Random(seed);
		else
			r = new Random();
		for (int i=0;i < arr.length;i++)
			arr.data[i] = r.nextFloat();
		return arr;
	}
	// mathematical operations.
	// addition
	public static Data add(Data d1, Data d2)
	{
		int[] sh=getCommonShape(d1.shape, d2.shape);
		Data res=new Data(sh).setRequiresGradient(d1.requiresGradient() | d2.requiresGradient());
		if (res.requiresGradient())
			res.setGradientFunction(GradFunc.additionGradient, d1, d2);
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
	public static Data add(Data d1, float d2)
	{
		int[] sh=d1.shape;
		Data res=new Data(sh);
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
	public static Data sub(Data d1, Data d2)
	{
		int[] sh=getCommonShape(d1.shape, d2.shape);
		Data res=new Data(sh);
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
	public static Data sub(Data d1, float d2)
	{
		int[] sh=d1.shape;
		Data res=new Data(sh);
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
	public static Data sub(float d1, Data d2)
	{
		int[] sh=d2.shape;
		Data res=new Data(sh);
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
	public static Data mul(Data d1, Data d2)
	{
		int[] sh=getCommonShape(d1.shape, d2.shape);
		Data res=new Data(sh);
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
	public static Data mul(Data d1, float d2)
	{
		int[] sh=d1.shape;
		Data res=new Data(sh);
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
	public static Data div(Data d1, Data d2)
	{
		int[] sh=getCommonShape(d1.shape, d2.shape);
		Data res=new Data(sh);
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
	public static Data div(Data d1, float d2)
	{
		int[] sh=d1.shape;
		Data res=new Data(sh);
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
	// thus functions is optional.
	public static Data div(float d1, Data d2)
	{
		int[] sh=d2.shape;
		Data res=new Data(sh);
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
	// dot product start.
	public static Data dot(Data d1, Data d2)
	{

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
		int[] newShape=getShapeForDot(d1.shape, d2.shape);
		// System.out.println("final shape :" + Arrays.toString(newShape));
		d1 = d1.as2DArray();
		d2 = d2.transpose(prepareAxisForDot(d2.shape.length)).reshape(d2.shape[d2.shape.length - 2], -1);
		// System.out.println(d2);
		int[]sh1=d1.shape;
		int[]sh2=d2.shape;
		if (sh1[1] != sh2[0])
			throw new RuntimeException("two matrixes doesn't match (" + sh1[1] + "," + sh2[0] + ")");
		int[] dotShape={sh1[0],sh2[1]};
		// System.out.println(".." + Arrays.toString(sh1) + ".." + Arrays.toString(sh2) + "..." + Arrays.toString(dotShape));
		float[] f=new float[sh1[0] * sh2[1]];

		for (int r=0;r < sh1[0];r++)
			for (int c=0;c < sh2[1];c++)
			{
				float sum=0;
				for (int i=0;i < sh1[1];i++)
					sum += d1.get(r, i) * d2.get(i, c);
				f[shapeToIndex(ar(r, c), dotShape)] = sum;
			}

		return new Data(f, dotShape).reshapeLocal(newShape);
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
	public static Data convolve1d(Data d1, Data kr)
	{
		// this convolve only support 1d, 1d kern.
		kr = kr.trim();
		if (kr.trim().shape.length != 1)
			throw new RuntimeException("convolve 1d error : expected 1d kernel array");
		d1 = d1.as2DArray();
		int len=d1.shape[1] - kr.shape[0] + 1;
		int[]outShape={d1.shape[0],len};
		float[] out=new float[d1.shape[0] * len];
		float[] tmpkr=new float[kr.shape[0]];
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
						sm += d1.get(dr, w + k) * kv; // get kernel in reverse order
						tmpkr[kp] = kv;
						kp--; // count downward
					}
					out[shapeToIndex(new int[]{dr,w}, outShape)] = sm;
				}
			}
			else
			{
				// convolve mode normal.
				// at the second row of the data, the tmp kernel will be cached so we can use it for faster iteration.
				for (int w=0;w < len;w++)
				{
					float sm=0;
					int kp=kr.shape[0] - 1;
					for (int k=0;k < kr.shape[0];k++)
					{
						sm += d1.get(dr, w + k) * tmpkr[kp]; // get kernel in reverse order
						kp--; // count downward
					}
					out[shapeToIndex(new int[]{dr,w}, outShape)] = sm;
				}
			}
		}
		return new Data(out, outShape);
	}
	public static Data correlate1d(Data d1, Data kr)
	{
		// this convolve, only support 1d kern.
		kr = kr.trim();
		if (kr.trim().shape.length != 1)
			throw new RuntimeException("correlation 1d error : expected 1d kernel array");
		d1 = d1.as2DArray();
		int len=d1.shape[1] - kr.shape[0] + 1;
		int[]outShape={d1.shape[0],len};
		float[] out=new float[d1.shape[0] * len];
		float[] tmpkr=new float[kr.shape[0]];
		for (int dr=0;dr < d1.shape[0];dr++)
		{
			// iterate over rows of input data.
			if (dr == 0)
			{
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
			else
			{
				// convolve mode normal.
				// at the second row of the data, the tmp kernel will be cached so we can use it for faster iteration.
				for (int w=0;w < len;w++)
				{
					float sm=0;
					for (int k=0;k < kr.shape[0];k++)
					{
						sm += d1.get(dr, w + k) * tmpkr[k]; // get kernel in reverse order
					}
					out[shapeToIndex(new int[]{dr,w}, outShape)] = sm;
				}
			}
		}
		return new Data(out, outShape);
	}
}
