package gss;

import java.util.*;

public class Util
{
	public static int shapeToIndex(int[]indSh, int[]shape, int[]strides)
	{
		// lazy indexing...
		int newPos=0;
		int strShPos = Math.max(0, shape.length - indSh.length);
		int strIndPos = Math.max(0, indSh.length - shape.length);
		for (int i = 0;i < Math.min(shape.length, indSh.length);i++)
		{
			int shps=i + strShPos;
			int shapeInd = Math.min(indSh[i + strIndPos], shape[shps] - 1); // uncomment to enable lazy broadcasting(value broadcasting).
			newPos += shapeInd *  strides[shps];
		}
		return newPos;
	}
	public static int shapeToIndex(int[]index, int[]shape)
	{
		int flatIndex=0;
		int stride = 1;
		// Calculate strides in reverse order (row-major)
		for (int i = shape.length - 1; i >= 0; i--)
		{
			flatIndex += index[i] * stride;
			stride *= shape[i];
		}
		return flatIndex;
	}
	public static int[] indexToShape(int index, int[]shape, int[]out)
	{
		for (int i=shape.length - 1;i >= 0;i--) // count down starts from shape.length -1 down to 0.
		{
			out[i] = index % shape[i];
			index = index / shape[i];
		}
		return out;
	}
	public static int length(int...sh)
	{
		int size=1;
		for (int s:sh)
		{
			if (s <= 0)
				throw new RuntimeException("shape with value of \"< 0\" doesn't allowed (" + Arrays.toString(sh) + ")");
			size *= s;
		}
		return size;
	}
	public static int[] genStrides(int[] shape)
	{
		int[]str = new int[shape.length];
		str[str.length - 1] = 1;
		int sm=1;
		for (int i=shape.length - 1;i >= 1;i--)
		{
			sm *= shape[i];
			str[i - 1] = sm;
		}
		return str;
	}
	public static int[] getCommonShape(int[] shape1, int[] shape2)
	{
		int[] newShape1=Arrays.copyOf(shape1.length > shape2.length ?shape1: shape2, Math.max(shape1.length, shape2.length));
		int[] newShape2=shape1.length > shape2.length ?shape2: shape1;
		for (int i=0;i < Math.min(shape1.length, shape2.length);i++)
		{
			int sh1=newShape1[newShape1.length - i - 1];
			int sh2=newShape2[newShape2.length - i - 1];
			if (sh1 != sh2 && (sh1 != 1 && sh2 != 1))
				throw new RuntimeException("not broadcastable shape at. ( " + sh1 + " != " + sh2 + " )");
			newShape1[newShape1.length - 1 - i] = sh1 == 1 ?sh2: sh1;
		}
		return newShape1;
	}
	public static int[] ar(int...a)
	{
		return a;
	}
	public static float[] flatten(float[][] data)
	{
		int rw=data.length;
		int cl=data[0].length;
		int pos=0;
		float[] dt=new float[rw * cl];
		for (int r=0;r < rw;r++)
			for (int c=0;c < cl;c++)
				dt[pos++] = data[r][c];
		return dt;
	}
	public static float[] flatten(float[][][] data)
	{
		int dp=data.length;
		int rw=data[0].length;
		int cl=data[0][0].length;
		int pos=0;
		float[] dt=new float[dp * rw * cl];
		for (int d=0;d < dp;d++)
			for (int r=0;r < rw;r++)
				for (int c=0;c < cl;c++)
					dt[pos++] = data[d][r][c];
		return dt;
	}
	public static float[] flatten(float[][][][] data)
	{
		int dp=data.length;
		int rw=data[0].length;
		int cl=data[0][0].length;
		int dd=data[0][0][0].length;
		int pos=0;
		float[] dt=new float[dp * rw * cl * dd];
		for (int d=0;d < dp;d++)
			for (int r=0;r < rw;r++)
				for (int c=0;c < cl;c++)
					for (int i=0;i < dd;i++)
						dt[pos++] = data[d][r][c][i];
		return dt;
	}
	public static int n(int[]sh, int n)
	{
		return sh[sh.length - n - 1];
	}
	public static String getString(String s, int count)
	{
		StringBuilder o=new StringBuilder();
		for (int i=0;i < count;i++)
			o.append(s);
		return o.toString();
	}
	public static String decString(String text, String decore, int length)
	{
		String pl=getString(decore, length);
		StringBuilder s=new StringBuilder(pl);
		s.append(" ");
		s.append(text);
		s.append(" ");
		s.append(pl);
		return s.toString();
	}
	public static String decString(String text, int length)
	{
		return decString(text, "+", length);
	}
	public static String line(int cnt)
	{
		StringBuilder sb=new StringBuilder();
		for (int i=0;i < cnt;i++)
			sb.append("-");
		return sb.toString();
	}
	public static float[] range(float len)
	{
		return range(0, len, 1);
	}
	public static float[] range(float str, float end)
	{
		return range(str, end, 1);
	}
	public static float[] range(float str, float end, float inc)
	{
		float cnt=(end - str);
		float[] f=new float[(int)(cnt / inc) + (cnt % inc == 0 ?0: 1)];
		int p=0;
		for (float i=str;i < end;i += inc)
		{
			f[p] = i;
			p++;
		}
		return f;
	}
	public static int[] range(int len)
	{
		return range(0, len, 1);
	}
	public static int[] range(int str, int end)
	{
		return range(str, end, 1);
	}
	public static int[] range(int str, int end, int inc)
	{
		int cnt=Math.abs(end - str);
		int[] arr=new int[(cnt / inc) + (cnt % inc == 0 ?0: 1)];
		inc = str > end ?-1: inc;
		if (inc > 0)
		{
			int p=0;
			for (int i=str;i < end;i += inc)
			{
				arr[p] = i;
				p++;
			}
		}
		else
		{
			int p=0;
			for (int i=str;i >= end;i += inc)
			{
				arr[p] = i;
				p++;
			}
		}
		return arr;
	}
	public static float[] asFloat(int...data)
	{
		float[] f=new float[data.length];
		for (int i=0;i < data.length;i++)
			f[i] = data[i];
		return f;
	}
	public static int[] fill(int[]sh, int len)
	{
		if (sh.length >= len)
			return sh;
		int[] s=new int[len];
		for (int i=0;i < s.length;i++)
			s[i] = i >= sh.length ?0: sh[i];
		return s;
	}
	public static int[][] fill(int[][] rng, int[] shp)
	{
		int[][] r=new int[shp.length][3];
		for (int i=0;i < shp.length;i++)
		{
			if (i < rng.length)
			{
				r[i][2] = rng[i][2];
				r[i][0] = rng[i][0];
				if (r[i][0] == -1)
					r[i][0] = 0;
				r[i][1] = rng[i][1];
				if (r[i][1] == -1)
					r[i][1] = shp[i];
			}
			else
			{
				r[i][0] = 0;
				r[i][1] = shp[i];
				r[i][2] = 1;
			}
		}
		return r;
	}
	public static int[] r(int end)
	{
		return new int[]{0,end,1};
	}
	public static int[] r(int start, int end)
	{
		return new int[]{start,end,1};
	}
	public static int[] r(int start, int end, int inc)
	{
		return new int[]{start,end,inc};
	}
	public static void print(Object...objs)
	{
		for (Object o:objs)
		{
			System.out.print(o);
			System.out.print(" ");
		}
		System.out.println();
	}
}
