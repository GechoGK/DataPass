package gss.arr;

import java.util.*;

import static gss.Util.*;

public class Data
{
	/*
	 improvement
	 in this class set and get methods are usefull.
	 and we can optimize the functions by making a method for a specific dimension.
	 */
	// for this data, (view, broadcast, reshape and transpose) is enough.
	public int[] shape;
	public int[] strides;
	public int length;
	public float[] data;

	public Data(int...shape)
	{
		this.shape = shape;
		this.strides = genStrides(shape);
		this.length = length(shape);
		this.data = new float[length];
	}
	public Data(float[] src)
	{
		this.data = src == null ?new float[0]: src;
		this.shape = new int[]{data.length};
		strides = new int[]{1};
		this.length = length(shape);
	}
	public Data(float[] src, int[]shp)
	{
		this.data = src == null ?new float[0]: src;
		this.shape = shp;
		strides = genStrides(shape);
		length = length(shape);
		if (length != data.length)
			throw new RuntimeException("invalid shape or data. they are not equal in length.");
	}
	public Data(float[] src, int[]shp, int[] strd)
	{
		this.data = src == null ?new float[0]: src;
		this.shape = shp;
		strides = strd;
		length = length(shape);
		if (length != data.length)
			throw new RuntimeException("invalid shape or data. they are not equal in length.");
	}
	public int shapeToIndex(int...index)
	{
		int newPos=0;
		int strShPos = Math.max(0, shape.length - index.length);
		int strIndPos = Math.max(0, index.length - shape.length);
		for (int i = 0;i < Math.min(shape.length, index.length);i++)
		{
			int shps=i + strShPos;
			int shapeInd = Math.min(index[i + strIndPos], shape[shps] - 1); // uncomment to enable lazy broadcasting(value broadcasting).
			newPos += shapeInd *  strides[shps];
		}
		return newPos; // + offset;
	}
	public int[] indexToShape(int index)
	{
		int[] out=new int[shape.length];
		for (int i=shape.length - 1;i >= 0;i--) // count down starts from shape.length -1 down to 0.
		{
			out[i] = index % shape[i];
			index = index / shape[i];
		}
		return out;
	}
	public float get(int...index)
	{
		int ind=Math.max(0, shapeToIndex(index));
		return data[ind];
	}
	// get array data length of @count.
	public float[] get(int[]index, int count)
	{
		int ind=shapeToIndex(index);
		if (count == -1)
			count = data.length - ind;
		float[]f=new float[count];
		int pos=0;
		for (int i=ind;i < ind + Math.min(data.length - ind, count);i++)
		{
			f[pos++] = data[i];
		}
		return f;
	}
	// set single value to the data.
	public void set(int[]index, float val)
	{
		int ind=shapeToIndex(index);
		data[ind] = val;
	}
	/*
	 // set multiple values(array valuea into the data
	 // @startIndex[] which start index of the destination data.
	 // @values[] floag value that is going to copied.
	 !! if the values array is greater than the destination array,
	 // it will write up the maximum data length, the rest will trimmed.

	 */
	public void set(int[]startIndex, float...values)
	{
		int ind=shapeToIndex(startIndex);
		int pos=0;
		for (int i=ind;i < ind + Math.min(values.length, data.length - ind);i++)
			data[i] = values[pos++];
	}
	// this method is the same as set Array but it repeates the same value upto the count range.
	public void set(int[]index, float val, int count)
	{
		int ind=shapeToIndex(index);
		if (count == -1)
			count = data.length - ind;
		for (int i=ind;i < ind + Math.min(count, data.length - ind);i++)
		{
			data[i] = val;
		}
	}
	// not tested.
	public float getRaw(int index)
	{
		return data[index];
	}
	public float[] getRaw(int index, int count)
	{
		float[] f=new float[count];
		for (int i=0;i < count;i++)
			f[i] = data[index + i];
		return f;
	}
	public float[] getRaw(int index, float[]out, int count)
	{
		float[] f=new float[count];
		for (int i=0;i < count;i++)
			f[i] = data[index + i];
		return f;
	}
	// not tested.
	public void setRaw(int index, float val)
	{
		data[index] = val;
	}
	// not tested.
	public void setRaw(int index, float[] dt)
	{
		int pos=0;
		for (int i=index;i < Math.min(dt.length, data.length - index);i++)
			data[index] = dt[pos++];
	}
	public Data reshape(int...newShape)
	{
		fillShape(newShape);
		if (isTransposed())
		{
			Data d=copy().reshapeLocal(newShape);
			return d;
		}
		if (length != length(newShape))
			throw new RuntimeException("can't view this array into " + Arrays.toString(newShape) + " shape. Reason!! the length is not equal");
		return new Data(data, newShape);
	}
	public Data reshapeLocal(int...newShape)
	{
		fillShape(newShape);
		if (isTransposed())
			throw new RuntimeException("unable to modify the shape of transposed array.");
		this.shape = newShape;
		strides = genStrides(shape);
		length = length(shape);
		if (length != data.length)
			throw new RuntimeException("invalid shape or data. they are not equal in length.");
		return this;
	}
	public Data transpose()
	{
		int[] ax=new int[shape.length];
		for (int i=0;i < ax.length;i++)
			ax[i] = ax.length - i - 1;
		return transpose(ax);
	}
	public Data transpose(int...axes)
	{
		if (axes.length != shape.length)
			throw new RuntimeException("invalid axes");
		int[] sh=new int[shape.length]; // Arrays.copyOf(shape, shape.length);
		int[] strd=new int[shape.length];
		int p=0;
		for (int i:axes)
		{
			if (i >= axes.length)
				throw new IndexOutOfBoundsException("index must not greater than the dimension o the array");
			sh[p] = shape[i];
			strd[p] = strides[i];
			p++;
		}
		return new Data(data, sh, strd);
	}
	public int[] fillShape(int...shp)
	{
		int nIndex=-1;
		for (int i=0;i < shp.length;i++)
		{
			if (shp[i] == -1 && nIndex != -1)
				throw new RuntimeException("the shape can't have multiple -1 values.");
			else if (shp[i] == -1)
				nIndex = i;
		}
		if (nIndex != -1)
		{
			shp[nIndex] = 1;
			int len=length(shp);
			int remSize=length / len;
			if (length % len != 0)
				throw new RuntimeException("choose an appropriate array size: unable to fill the missing value.");
			shp[nIndex] = remSize;
		}
		return shp;
	}
	public boolean isBrodcastable(int[]newShape)
	{
		if (newShape.length < shape.length)
			return false;
		int len=newShape.length - shape.length;
		for (int i=0;i < shape.length;i++)
		{
			if (!(shape[i] == newShape[len + i] || (shape[i] == 1 && newShape[len + i] > 0)))
				return false;
		}
		return true;
	}
	public boolean isTransposed()
	{
		int[] strd=genStrides(shape);
		return !Arrays.equals(strd, strides);
	}
	public Data copy()
	{
		if (isTransposed()) // or broadcasted.
		{
			float[] dt=new float[length];
			for (int i=0;i < length;i++)
			{
				int[] shp=indexToShape(i);
				dt[i] = get(shp);
			}
			return new Data(dt, shape);
		}
		else
		{
			return new Data(Arrays.copyOf(data, data.length), shape);
		}
	}
	// not tested
	// needs some improvements.
	// it must support broadcasting.
	// the new shape can be a broadcast if the original shape.
	// the reshape method can't give that ability.
	public Data copyAs2(int...newShape)
	{
		if (Arrays.equals(newShape, shape))
			return copy();
		if (isBrodcastable(newShape))
		{
			int len=length(newShape);
			float[] dt=new float[len];
			for (int i=0;i < len;i++)
			{
				int[] shp=indexToShape(i);
				dt[i] = get(shp);
			}
			return new Data(dt, newShape);
		}
		return copy().reshapeLocal(newShape);
	}
	@Override
	public String toString()
	{
		String inf="shape : " + Arrays.toString(shape) + " len :" + length + (isTransposed() ?" : transposed.": ".");
		return inf;
	}
	public void printArray()
	{
		int dm=shape.length;
		int[] cShape=new int[dm];
		StringBuilder sb=new StringBuilder();
		printArray2(cShape, 0, sb);
		String s=sb.toString();
		s = s.replace("],", "],\n");
		System.out.println(s);
	}
	private void printArray2(int[]sh, int dm, StringBuilder sb)
	{
		if (sh.length == dm)
		{
			sb.append(get(sh));
			return;// "" + get(sh);
		}
		sb.append("[");
		for (int i = 0; i < shape[dm]; i++)
		{
			if (i > 0)
			{
				sb.append(", "); // Separator between elements
			}
			sh[dm] = i; // Set current dimension index
			printArray2(sh, dm + 1, sb); // Recurse to next dimension
		}
		sb.append("]");
	}
	public Data as1DArray()
	{
		return reshape(-1);
	}
	public Data as2DArray()
	{
		return reshape(-1, shape[shape.length - 1]);
	}
	public Data as3DArray()
	{
		if (shape.length == 1)
		{
			return reshape(-1, 1, shape[shape.length - 1]);
		}
		return reshape(-1, shape[shape.length - 2], shape[shape.length - 1]);
	}
	public Data trim()
	{
		int c=0;
		for (int i=0;i < shape.length;i++)
		{
			if (shape[i] == 1)
				c++;
		}
		if (c == 0)
		{
			// no change
			return this;
		}
		int[] sh=Arrays.copyOfRange(shape, c, shape.length);
		return new Data(data, sh, strides);
	}
}

