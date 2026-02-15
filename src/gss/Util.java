package gss;

import gss.arr.*;
import java.io.*;
import java.util.*;

import static gss.Functions.*;

public class Util
{
	/*
	 shapeToIndex(...);
	 which converts the shape of an array into flat index.
	 strides used for arrays like transposed and sliced types.
	 @int[] indShape 	whape we want to indexed.
	 @int[] shape 		the actual array shape to loop from.
	 @int[] strides 	used to calculate the length.

	 */
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
	/*
	 shapeToIndex(...);
	 which converts the shape of an array into flat index.
	 strides calculated om the way.

	 @int[] indShape 	whape we want to indexed.
	 @int[] shape 		the actual array shape to loop from.

	 */
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
	/*
	 convert a flat index into a desired shape based on the gives array shape.

	 @int index. the index want to convert into shape.
	 @int[] shape. the actual maximum shape.
	 */
	public static int[] indexToShape(int index, int[]shape)
	{
		return indexToShape(index, shape, new int[shape.length]);
	}
	/*
	 the same as the above, except it expects @int[] out,
	 which returns the calculated shape from flat index.
	 */
	public static int[] indexToShape(int index, int[]shape, int[]out)
	{
		for (int i=shape.length - 1;i >= 0;i--) // count down starts from shape.length -1 down to 0.
		{
			out[i] = index % shape[i];
			index = index / shape[i];
		}
		return out;
	}
	/*
	 calculate the length of an array from the given shape.
	 @usage
	 length(2,3,4) = 24
	 */
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
	/*
	 generate strides from the given shape.
	 @int[] shape

	 -- usage
	 genStrides(2,3,4) = [12,4,1]
	 */
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
	/*
	 this function returns the shape from two shapes.
	 both shapes must be broadcastable each other.
	 mostly it takes the large shape, because the smaller one can be broadcasted into the larger shape.
	 @int[] shape1, shape1
	 @int[] shape2, shape

	 -- usage
	 getCommonShape(new int[]{2,1,3},new int[]{5,2,8,3});
	 == [5,2,8,3]
	 */
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
	public static int[] getCommonShapeExcept(int[] shape1, int[] shape2, int axis)
	{
		int[] newShape1=Arrays.copyOf(shape1.length > shape2.length ?shape1: shape2, Math.max(shape1.length, shape2.length));
		int[] newShape2=shape1.length > shape2.length ?shape2: shape1;
		int len=Math.min(shape1.length, shape2.length);
		int maxLen=Math.max(shape1.length, shape2.length) - 1;
		for (int i=0;i < len;i++)
		{
			int idx=maxLen - i;
			if (axis == idx)
				continue;
			int sh1=newShape1[newShape1.length - i - 1];
			int sh2=newShape2[newShape2.length - i - 1];
			if (sh1 != sh2 && (sh1 != 1 && sh2 != 1))
				throw new RuntimeException("not broadcastable shape at. ( " + sh1 + " != " + sh2 + " )");
			newShape1[newShape1.length - 1 - i] = sh1 == 1 ?sh2: sh1;
		}
		return newShape1;
	}
	/*
	 generate an array from input.
	 it helps to reduce the java "new int[]{}",
	 we can just use ar(1,2,3)
	 == [1,2,3]
	 @int...a input array

	 -- usage ar(2,3,4)
	 */
	public static int[] ar(int...a)
	{
		return a;
	}
	/*
	 flatten the input 2d array into 1d array.
	 @float[][] data

	 -- usage
	 flatten(new int[][]{new int[]{2,3},new int[]{4,5}});
	 ==[2,3,3,4]
	 */
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
	// convert 3d array into 1d array.
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
	// convert 4d array into 1d array.
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
	/*
	 returns the value from the end to the start.
	 @int[] sh, input shape or other int arrays.
	 @int n, value from the end.

	 --usage
	 int[] a=new int[]{2,3,4};
	 n(a,0); == 4
	 n(a,1); == 3
	 n(a,2); == 2
	 n(a,3)) == out of range error.
	 */
	public static int n(int[]sh, int n)
	{
		return sh[sh.length - n - 1];
	}
	/*
	 generate a string repratedly
	 @string a intput string
	 @int cout the number which we want to repeat.

	 -- usage
	 getString("a",5) = "aaaaa"
	 */
	public static String getString(String s, int count)
	{
		StringBuilder o=new StringBuilder();
		for (int i=0;i < count;i++)
			o.append(s);
		return o.toString();
	}
	/*
	 decorate a string, debug purpose.
	 it addes a decorating pattern at the beginning and end of a string.
	 @string text the text which will be decorated.
	 @string decore decoring text.
	 @int length the length of the devore to be repeated.

	 -- usage
	 decString("hello","-",5);
	 == "----- hello -----"
	 */
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
	/*
	 it decores a string the same as above.
	 eccept it uses default ("+") decoring value.

	 --usage
	 decString("hello",5);
	 == "+++++ hello +++++"
	 */
	public static String decString(String text, int length)
	{
		return decString(text, "+", length);
	}
	/*
	 generate a repeating "-" pattern @cnt times
	 @int cnt number of "-" to be repeated.

	 -- usage
	 line(7);
	 == "-------";
	 */
	public static String line(int cnt)
	{
		StringBuilder sb=new StringBuilder();
		for (int i=0;i < cnt;i++)
			sb.append("-");
		return sb.toString();
	}
	/*
	 generate a float array within range(end) but not included end
	 from 0 to end with increment 1.

	 @float end. the range wich array ends but not included.

	 --usage
	 range(5);
	 == [0,1,2,3,4];

	 */
	public static float[] range(float end)
	{
		return range(0, end, 1);
	}
	/*
	 generate a float array within range(start to end) but not included end
	 from start(str) to end(end) with increment 1.

	 @float str. the range wich array starts
	 @float end. the range chich the array ends but not included.
	 --usage
	 range(3,8);
	 == [3,4,5,6,7];

	 */
	public static float[] range(float str, float end)
	{
		return range(str, end, 1);
	}
	/*
	 generate a float array within range(start to end) but not included end with increment (inc).
	 from start(str) to end(end) with increment inc.

	 @float str. the range wich array starts
	 @float end. the range chich the array ends but not included.
	 @float inc. increment value.
	 --usage
	 range(5,15,2);
	 == [5,7,9,11,13];

	 range(5,8,0.5f)
	 == [5.0, 5.5, 6.0, 6.5, 7.0, 7.5); 
	 */
	public static float[] range(float str, float end, float inc)
	{
		float cnt=(end - str);
		float[] f=new float[(int)(cnt / inc)];// + (cnt % inc == 0 ?0: 1)];
		int p=0;
		for (int i=0;i < f.length;i++)
			f[p++] = (i * inc) + str;
//		for (float i=str;i < end;i += inc)
//		{
//			f[p] = i;
//			p++;
//		}
		return f;
	}
	/*
	 generate a int array within range(end) but not included end
	 from 0 to end with increment 1.

	 @int end. the range wich array ends but not included.

	 --usage
	 range(5);
	 == [0,1,2,3,4];

	 */
	public static int[] range(int len)
	{
		return range(0, len, 1);
	}
	/*
	 generate a int array within range(start to end) but not included end
	 from start(str) to end(end) with increment 1.

	 @int str. the range wich array starts
	 @int end. the range chich the array ends but not included.
	 --usage
	 range(3,8);
	 == [3,4,5,6,7];

	 */
	public static int[] range(int str, int end)
	{
		return range(str, end, 1);
	}
	/*
	 generate a float array within range(start to end) but not included end with increment (inc).
	 from start(str) to end(end) with increment inc.

	 @int str. the range wich array starts
	 @int end. the range chich the array ends but not included.
	 @imt inc. increment value.
	 --usage
	 range(5,15,2);
	 == [5,7,9,11,13];

	 */
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
	// convert int array into float array.
	public static float[] asFloat(int...data)
	{
		float[] f=new float[data.length];
		for (int i=0;i < data.length;i++)
			f[i] = data[i];
		return f;
	}
	public static float[]asFloat(Float...d)
	{
		float[] f=new float[d.length];
		for (int i=0;i < f.length;i++)
			f[i] = d[i];
		return f;
	}
	/*
	 this function serves as a python range [::]
	 @int[][] rng. range
	 @int[] shp. shape of an array.

	 --usage
	 int[] r1={5}; // end =5
	 int[] r2={3};
	 range(new int[]{r1, r2},new int[]{5,10});
	 == [[0,5,1], [0,3,1]] = [start, end, increment];

	 int[] r1={2,5}; // start =2, end = 5
	 int[] r2={0,3};
	 range(new int[]{r1, r2},new int[]{5,10});
	 == [[2,5,1], [0,3,1]] = [start, end, increment];

	 int[] r1={0,5,2}; // start = 0, end = 5, inc = 2;
	 int[] r2={1,10,3};
	 range(new int[]{r1, r2},new int[]{5,10});
	 == [[0,5,2], [1,10,3]] = [start, end, increment];

	 int[] r1={2,-1}; // start = 2, end = end of array(5)
	 int[] r2={-1,7}; // start = start of array 0, end = 7;
	 range(new int[]{r1, r2},new int[]{5,10});
	 == [[2,5,1], [0,7,1]] = [start, end, increment];


	 */
	public static int[][] fill(int[][] rng, int[] shp)
	{
		int[][] r=new int[shp.length][3];
		for (int i=0;i < shp.length;i++)
		{
			if (i < rng.length)
			{
				if (rng[i].length == 0)
				{
					// range not supplied fill by default.
					r[i][0] = 0; // start. default 0.
					r[i][1] = shp[i]; // end. default value in shape at current index.
					r[i][2] = 1; // increment. default 1.
				}
				else if (rng[i].length == 1)
				{
					// only end value is found. fill the rest.
					r[i][0] = 0; // start. fill 0 for default value.
					r[i][1] = rng[i][0] == -1 ?shp[i]: rng[i][0]; // end
					r[i][2] = 1; // increment. default 1.
				}
				else if (rng[i].length == 2)
				{
					// only 2 values are found(start and end)
					r[i][0] = rng[i][0] == -1 ?0: rng[i][0]; // start
					r[i][1] = rng[i][1] == -1 ?shp[i]: rng[i][1]; // end
					r[i][2] = 1; // increment. for default 1.
				}
				else if (rng[i].length == 3)
				{
					// all 3(start, end, increment) values are found
					r[i][0] = rng[i][0] == -1 ?0: rng[i][0]; // start
					r[i][1] = rng[i][1] == -1 ?shp[i]: rng[i][1]; // end
					r[i][2] = rng[i][2]; // increment. (-)values are not allowed.
				}
				else error("unknown index range (" + rng[i].length + ")");
			}
			else
			{
				// range not supplied fill by default.
				r[i][0] = 0; // start. default 0.
				r[i][1] = shp[i]; // end. default value in shape at current index.
				r[i][2] = 1; // increment. default 1.
			}
		}
		return r;
	}
	/*
	 generate range array used for slicing NDArray.
	 @int end, end of array

	 -- usage
	 r(5); = [0,5,1];
	 */
	public static int[] r(int end)
	{
		return new int[]{0,end,1};
	}
	/*
	 generate range array used for slicing NDArray.
	 @int start, start of an array.
	 @int end, end of array

	 -- usage
	 r(2, 5); = [2,5,1];
	 */
	public static int[] r(int start, int end)
	{
		return new int[]{start,end,1};
	}
	/*
	 generate range array used for slicing NDArray.
	 @int start, start of an array.
	 @int end, end of array
	 @int inc, increment valu fo array.

	 -- usage
	 r(3, 7, 2); = [3, 7, 2];

	 Base b=NDArray.range(50).reshape(10, 5);

	 Base sliced= b.slice(new int[][]{r(2)});

	 = [10, 11, 12, 13, 14]

	 */
	public static int[] r(int start, int end, int inc)
	{
		return new int[]{start,end,inc};
	}
	/*
	 print objects and arrays in an new line.
	 */
	public static void println(Object...objs)
	{
		printO(true, objs);
	}
	/*
	 print objects and arrays.
	 */
	public static void print(Object...objs)
	{
		printO(false, objs);
	}
	// internal method of print.
	public static void printO(boolean newLn, Object...objs)
	{
		for (Object o:objs)
		{
			if (o instanceof int[])
				System.out.print(Arrays.toString((int[])o));
			else if (o instanceof float[])
				System.out.print(Arrays.toString((float[])o));
			else if (o instanceof char[])
				System.out.print(Arrays.toString((char[])o));
			else if (o instanceof double[])
				System.out.print(Arrays.toString((double[])o));
			else if (o instanceof long[])
				System.out.print(Arrays.toString((long[])o));
			else if (o instanceof short[])
				System.out.print(Arrays.toString((short[])o));
			else if (o instanceof byte[])
				System.out.print(Arrays.toString((byte[])o));
			else if (o instanceof boolean[])
				System.out.print(Arrays.toString((boolean[])o));
			else if (o instanceof Object[])
			{
				System.out.print(Arrays.toString((Object[])o));
			}
			else if (o instanceof ArrayList)
			{
				for (int i=0;i < ((ArrayList)o).size();i++)
					printO(newLn, ((ArrayList)o).get(i));
			}
			else
				System.out.print(o);
			System.out.print(newLn ?"\n": " ");
		}
		System.out.println();
	}
	// reshape the shape an array into a new shape.
	public static int[] reshape(int[]newShape, int[]oldShape)
	{
		int nIndex=-1;
		for (int i=0;i < newShape.length;i++)
		{
			if (newShape[i] == -1 && nIndex != -1)
				throw new RuntimeException("the shape can't have multiple -1 values.");
			else if (newShape[i] == -1)
				nIndex = i;
		}
		if (nIndex != -1)
		{
			int length=length(oldShape);
			newShape[nIndex] = 1;
			int len=length(newShape);
			int remSize=length / len;
			if (length % len != 0)
				throw new RuntimeException("choose an appropriate array size: unable to fill the missing value.");
			newShape[nIndex] = remSize;
		}
		return newShape;
	}
	// transpose the shape in different axis.
	public static int[] transpose(int[]shape, int...axes)
	{
		if (axes.length != shape.length)
			throw new RuntimeException("invalid axes");
		int[] sh=new int[shape.length]; // Arrays.copyOf(shape, shape.length);
		int p=0;
		for (int i:axes)
		{
			if (i >= axes.length)
				throw new IndexOutOfBoundsException("index must not greater than the dimension o the array");
			sh[p] = shape[i];
			p++;
		}
		return sh;
	}
	public static boolean equals(Base b1, Base b2)
	{
		return equals(b1, b2, false);
	}
	public static boolean equals(Base b1, Base b2, boolean checkGrad)
	{
		if (!Arrays.equals(b1.shape, b2.shape))
			return false;
		for (int i=0;i < b1.length;i++)
		{
			int ind[]=indexToShape(i, b1.shape);
			if (b1.get(ind) != b2.get(ind))
				return false;
			if (checkGrad)
			{
				if (b1.getGrad(ind) != b2.getGrad(ind))
					return false;
			}
		}
		return true;
	}
	/*
	 this function checks whether two array classes are the same or not.
	 it have a gap for floating point precision.
	 @Base b1. first array class
	 @Base b2. second array class.
	 returns b1±thresh==b2

	 */
	public static boolean isClose(Base b1, Base b2)
	{
		return isClose(b1, b2, 0.001f);
	}
	public static boolean isClose(Base b1, Base b2, float thresh)
	{
		if (!Arrays.equals(b1.shape, b2.shape))
			return false;
		for (int i=0;i < b1.length;i++)
		{
			int ind[]=indexToShape(i, b1.shape);
			float v1=b1.get(ind);
			float v2=b2.get(ind);
			boolean close=(v1 + thresh >= v2 && v1 - thresh <= v2);
			if (!close)
				return false;
		}
		return true;
	}
	// throw an error.
	public static void error(Object o)
	{
		throw new RuntimeException(o + "");
	}
	public static void warn(Object o)
	{
		System.out.println("!!! " + o);
	}
	public static int[] copyB(int[] src, int len)
	{
		int[] out=new int[len];
		return copyB(src, len, out);
	}
	public static int[] copyB(int[] src, int[]out)
	{
		if (out == null)
			error("destination array can't be null");
		return copyB(src, out.length, out);
	}
	public static int[] copyB(int[] src, int len, int[] out)
	{
		if (out == null)
			out = new int[len];
		int l=src.length - 1;
		for (int i=len - 1;i >= 0;i--)
		{
			out[i] = src[l];
			l--;
			if (l < 0)
				break;
		}
		return out;
	}
	public static float[] copy(float[]src)
	{
		return Arrays.copyOf(src, src.length);
	}
	public static int[] copy(int[]src)
	{
		return Arrays.copyOf(src, src.length);
	}
	// new functions
	/*
	 remove item from primitive array.
	 @int[] arr. input array.
	 @int ind. index which will be removed.

	 -- usage.
	 int[] a={1,2,3};
	 removeAtIndex(a,1);
	 == [1,3];
	 */
	public static int[] remove(int[]arr, int ind)
	{
		if (ind >= arr.length)
			error("invalid index range(" + ind + ")");
		int[] newArr=new int[arr.length - 1];
		int index=0;
		for (int i=0;i < arr.length;i++)
		{
			if (i == ind)
				continue;
			newArr[index++] = arr[i];
		}
		return newArr;
	}
	public static int[]remove(int[]arr, int...ind)
	{
		// bloated.
		if (arr.length < ind.length)
			error("index length should not be grrate than array size");
		int[] out=new int[arr.length - ind.length];
		Arrays.sort(ind);
		int index=0;
		int indInd=0;
		for (int i=0;i < arr.length;i++)
		{
			if (indInd < ind.length)
			{
				if (i == ind[indInd])
				{
					indInd++;
					continue;
				}
				if (ind[indInd] < i)
					error("repeating index isn't allowed \"" + ind[indInd] + "\"");
			}
			out[index++] = arr[i];
		}
		return out;
	}
	/*
	 put new element into an array.
	 @int[] arr. target array.
	 @int val. value to be inserted.
	 @int ind. an index whih the value will be placed.

	 -- usage
	 int[] a={1,2};
	 putAtIndex(a,1,5);
	 == [1,5,2]
	 */
	public static int[] insert(int[]arr, int val, int ind)
	{
		if (ind < 0 || ind >= arr.length + 1)
			error("invalid index range(" + ind + ")");
		int[] newArr=new int[arr.length + 1];
		int index=0;
		for (int i=0;i < newArr.length;i++)
		{
			if (i == ind)
				continue;
			newArr[i] =  arr[index++];
		}
		newArr[ind] = val;
		return newArr;
	}
	public static int[] append(int[] arr, int val)
	{
		return insert(arr, val, arr.length);
	}
	public static int[]replace(int[]src, int index, int repV)
	{
		if (index >= src.length)
			error("the index should be less than the source array length. index(" + index + ") >= src.length(" + src.length + ")");
		src[index] = repV;
		return src;
	}
	public static int[]replace(int[]arr, int[]index, int[]val)
	{
		for (int i=0;i < index.length;i++)
			arr[index[i]] = val[i];
		return arr;
	}
	/*
	 loop through all posible values fro the given shape(array)
	 @int[] shape. the array to be iterated.

	 -- usage
	 int[] a={2,3}
	 loop(a);
	 == [0,0];
	 == [0,1]
	 == [0,2]
	 == [1,0]
	 == [1,1]
	 == [1,2]
	 it call the func.apply(...);
	 */
	public static float[] loop(int[]shapeOrig, ArrayFunction func)
	{
		return loop(shapeOrig, new int[0], func);
	}
	public static float[] loop(int[]shapeOrig, int[]axis, ArrayFunction func)
	{
		if (axis == null)
			axis = new int[0];
		if (axis.length > shapeOrig.length)
			error("axis length can't be greater than shape length, axis length :" + axis.length + " > " + shapeOrig.length);
		int[]shape=Arrays.copyOf(shapeOrig, shapeOrig.length);
		shape = fromNonAxis(shape, axis);
		int len=length(shape);
		float[]out=new float[len];
		for (int i=0;i < len;i++)
			out[i] = func.apply(indexToShape(i, shape));
		return out;
	}
	public static int[] loop(int[]shapeOrig, int[]axis, ArrayConsumer func)
	{
		if (axis == null)
			axis = new int[0];
		if (axis.length > shapeOrig.length)
			error("axis length can't be greater than shape length, axis length :" + axis.length + " > " + shapeOrig.length);
		int[]shape=Arrays.copyOf(shapeOrig, shapeOrig.length);
		for (int i=0;i < axis.length;i++)
			shape[axis[i]] = 1;
		int len=length(shape);
		print("original shape", shapeOrig, "loop over shape", shape, "axis", axis);
		for (int i=0;i < len;i++)
			func.consume(indexToShape(i, shape));
		return shape;
	}
	public static void loop(Base b, ArrayConsumer func)
	{
		int len=length(b.shape);
		for (int i=0;i < len;i++)
			func.consume(b.indexToShape(i));
	}
	// each index items should not be greater than arrays(arr).length.
	public static int[] collect(int[] arr, int[]index)
	{
		int[] out=new int[index.length];
		for (int i=0;i < index.length;i++)
			out[i] = arr[index[i]];
		return out;
	}
	public static int[] fromNonAxis(int[]sh, int[]ax)
	{
		for (int i=0;i < ax.length;i++)
			sh[ax[i]] = 1;
		return sh;
	}
	/*
	 read text from file.
	 readString(@String path)
	 @String path. location of the file.
	 location can be vary in different oprating systems. so consider that.

	 -- usage
	 !! let's say i have a file in internal storage called "a.txt" and it's content is "hello world".
	 readString("/sdcard/a.txt");
	 == "hello world"
	 */
	public static String readString(String path) throws IOException
	{
		FileInputStream fis=new FileInputStream(path);
		byte[] b=new byte[fis.available()];
		fis.read(b);
		fis.close();
		return new String(b);
	}
	/*
	 save String content to file.
	 @String path. drstination file.
	 @String cont. the content to be written.

	 -- usage
	 String path="/sdcard/a.txt";
	 String content="Hello world!";
	 saveString(path,content);
	 */
	public static void saveString(String path, String cont) throws Exception
	{
		FileOutputStream fos=new FileOutputStream(path);
		fos.write(cont.getBytes());
		fos.flush();
		fos.close();
	}
	/*
	 save Serializable Object to file.
	 @String path. drstination file.
	 @Object obj. the sedualizable object to be written.

	 -- usage
	 String path="/sdcard/a.txt";

	 Object content="Hello world!";
	 // or
	 content=24;
	 // or
	 content=468.7;
	 saveObject(path,content);
	 */
	public static void saveObject(String path, Object obj) throws Exception
	{
		FileOutputStream fos= new FileOutputStream(path);
		new ObjectOutputStream(fos).writeObject(obj);
		fos.close();
	}
	public static Object loadObject(String path) throws Exception
	{
		FileInputStream fis=new FileInputStream(path);
		Object obj= new ObjectInputStream(fis).readObject();
		fis.close();
		return obj;
	}
	public static String check(boolean b)
	{
		if (!b)
			error("assertion error");
		return "passed ✓";
	}
	public static boolean equalsExcept(int[] sh1, int[] sh2, int index)
	{
		if (sh1.length != sh2.length)
			return false;
		for (int i=0;i < sh1.length;i++)
		{
			if (i == index)
				continue;
			if (sh1[i] != sh2[i])
				return false;
		}
		return true;
	}
	public static int[] concatShape(int[] sh1, int[]sh2, int axis)
	{
		if (!equalsExcept(sh1, sh2, axis))
			error("can't concatinate the two shapes togather, they are not equal in shape other than concatenation axis");
		int[] sh=new int[sh1.length];
		for (int i=0;i < sh.length;i++)
		{
			sh[i] = sh1[i];
			if (i == axis)
				sh[i] = sh1[i] + sh2[i];
		}
		return sh;
	}
	// same as getCommonShape(int[] a, int[] b))
	public static int[]broadcast(int[]sh1, int[]sh2)
	{
		return getCommonShape(sh1, sh2);
	}
	public static int[]replaceValue(int[]src, int tarV, int repV)
	{
		for (int i=0;i < src.length;i++)
			if (src[i] == tarV)
				src[i] = repV;
		return src;
	}
	public static Scanner input(String...msg)
	{
		if (msg.length != 0)
			System.out.print(msg[0] + " >> ");
		else System.out.print(">> ");
		return new Scanner(System.in);
	}
	public static float[][] findDiff(float[][] a, float[][] b)
	{
		if (a.length != b.length)
			error("array a and b are not the same shape");
		List<float[]>o=new ArrayList<>();
		for (int r=0;r < a.length;r++)
			for (int c=0;c < a[0].length;c++)
				if (a[r][c] != b[r][c])
					o.add(new float[]{a[r][c], b[r][c],a[r][c] - b[r][c]});
		return o.toArray(new float[0][]);
	}
}
