package gss.arr;

import gss.*;
import java.util.*;

import static gss.Util.*;
import static gss.arr.GradFunc.*;

public class Base
{
	/*
	 improvement
	 in this class set and get methods are usefull.
	 and we can optimize the functions by making a method for a specific dimension.
	 */
	// for this data, (view, broadcast, reshape and transpose) is enough.
	/*
	 there are mothods that use axes to di their function.
	 when that happens, to iterate over a specific axis use strides.
	 example.
	 a = [[1,2,3],[4,5,6]];
	 a.shape = [2,3]
	 a.strides = [3,1];
	 axes[0] jumps using stride[0] = 3.
	 for(int i=0;i<a.shape[0];i++)
	 a[strides[0 * i] = 5;

	 >>  print(a)
	 >> [[5,2,3],[5,5,6]];
	 //////// !!!!! ///////
	 this stride also used for slicing. 
	 example.
	 a = [[1,2,3],[4,5,6]];
	 a.shape = [2,3]
	 a.strides = [3,1];
	 // now let's slice into othet dim.
	 b = a[:1]
	 >> b
	 >> [2,5]
	 >> b.shape => [2]
	 >> b.strides => [3]
	 >> offset => 1.
	 */
	// new methods to add.
	/*
	 public Data get(int...index){
	 return ...;
	 }
	 public void set(Data d,int...index){
	 ....
	 }
	 public Data slice(Range...ranges){
	 return ...;
	 }
	 */
	/*
	 note:
	 // avoid shape manipulation wuthout creating a new Base instance.
	 // in that process the gradient tracker will lost it's computation graph.
	 so, if you are using gradients and changing the shape of the Base class without new instance, please don't use these function
	 example.
	 -- reshapeLocal. that doesn't return a new Base class....
	 */
	public int[] shape;
	public int[] strides;
	public int length;
	public int offset=0;
	public Data data;
	public List<Base> childs = new ArrayList<>();
	public Object params=null;
	public GradFunc gradientFunction;

	public Base(int...shape)
	{
		this.shape = shape;
		this.strides = genStrides(shape);
		this.length = length(shape);
		this.data = new Data(shape);
	}
	public Base(float[] src)
	{
		this.data = new Data(src);
		this.shape = new int[]{data.length};
		strides = new int[]{1};
		this.length = length(shape);
	}
	public Base(float[] src, int...shp)
	{
		this.data = new Data(src);
		this.shape = shp;
		strides = genStrides(shape);
		length = length(shape);
		if (length != data.length)
			throw new RuntimeException("invalid shape or data. they are not equal in length.");
	}
	public Base(float[] src, int[]shp, int[] strd, int off)
	{
		this.data = new Data(src);
		this.shape = shp;
		strides = strd;
		this.offset = off;
		length = length(shape);
		// if (length != data.length)
		//	throw new RuntimeException("invalid shape or data. they are not equal in length.");
	}
	public Base(Data d, int[]shp, int off)
	{
		this.data = d;
		this.shape = shp;
		this.strides = genStrides(shape);
		this.length = length(shape);
		this.offset = off;
		// if (length != data.length)
		// 	throw new RuntimeException("invalid shape or data. they are not equal in length.");
	}
	public Base(Data d, int[]shp, int[]strd, int off)
	{
		this.data = d;
		this.shape = shp;
		strides = strd;
		length = length(shape);
		offset = off;
		// if (length != data.length)
		// 	throw new RuntimeException("invalid shape or data. they are not equal in length.");
	}
	public int getDim()
	{
		return shape.length;
	}
	// gradient area.
	public void backward()
	{
		if (gradientFunction == null)
			return;
		// throw new RuntimeException("gradient function not found = " + gradientFunction);
		// System.out.println("backward " + gradientFunction);
		gradientFunction.backward(this, childs.toArray(new Base[0]), params);
		for (Base arr:childs)
		{
			// System.out.println("=== " + arr.gradientFunction + " == " + this);
			arr.backward();
		}
	}
	public Base setRequiresGradient(boolean enableGrad)
	{
		data.setGradientEnabled(enableGrad);
		return this;
	}
	public boolean requiresGradient()
	{
		return data.requiresGradient;
	}
	public Base setGradientFunction(GradFunc func, Base...chlds)
	{
		this.gradientFunction = func;
		this.childs.clear();
		for (Base ar:chlds)
			this.childs.add(ar);
		return this;
	}
	public void setGradientParams(Object prms)
	{
		params = prms;
	}
	public Base detachGradient()
	{
		Base d=new Base(data.gradient, shape, strides, offset);
		return d;
	}
	public float getRawGrad(int ind)
	{
		return data.gradient[ind];
	}
	public float getGrad(int...index)
	{
		int ind=Math.max(0, shapeToIndex(index));
		return data.gradient[ind];
	}
	public void setGrad(int[]index, float val)
	{
		int ind=shapeToIndex(index);
		data.gradient[ind] += val;
	}
	public void setRawGrad(int ind, float v)
	{
		data.gradient[ind] += v;
	}
	public void fillGrad(float v)
	{
		Arrays.fill(data.gradient, v);
	}
	public void zeroGrad()
	{
		data.zeroGradient();
	}
	// end gradients.
	public Base slice(int...ind)
	{
		if (ind.length > shape.length)
			throw new RuntimeException("index.length must not be greater than shape.length:(" + Arrays.toString(ind) + ", " + Arrays.toString(shape) + ")");
		int off=shapeToIndex(Util.fill(ind, shape.length));
		int shl=shape.length - ind.length;
		int[] sh=null;
		int[] str=null;
		if (shl == 0)
		{
			sh = new int[]{1};
			str = new int[]{1};
		}
		else
		{
			sh = new int[shl];
			str = new int[shl];
			for (int i=0;i < sh.length;i++)
			{
				sh[i] = shape[ind.length + i];
				str[i] = strides[ind.length + i];
			}
			// System.out.println("offset = " + off + ", new shape :" + Arrays.toString(sh) + ", new stride :" + Arrays.toString(str));
		}
		return new Base(data, sh, str, off);
	}
	public Base slice(int[]...ind)
	{
		// need some improvement.
		if (ind.length > shape.length)
			throw new RuntimeException("index.length must not be greater than shape.length.");
		// int shl=shape.length - ind.length;
		// if (shl == 0)
		//	throw new RuntimeException("invalid index for slice.");
		// System.out.println("slicing with range :");
		int[] sh= new int[shape.length];
		int[] str=new int[shape.length];
		ind = Util.fill(ind, shape);
		// System.out.println(ind.length + ",, " + sh.length);
		int i=0;
		int off=0;
		for (int[] r:ind)
		{
			int ln=0;
			for (int k=r[0];k < r[1];k += r[2])
				ln++;
			sh[i] = ln;
			str[i] = strides[i] * r[2];
			off += r[0] * strides[i];
			i++;
		}
		// System.out.println("offset = " + off + ", new shape :" + Arrays.toString(sh) + ", new stride :" + Arrays.toString(str));
		return new Base(data, sh, str, off);
	}
	public void fill(float v)
	{
		Arrays.fill(data.items, v);
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
		return newPos + offset;
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
		return data.items[ind];
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
			f[pos++] = data.items[i];
		}
		return f;
	}
	// set single value to the data.
	public void set(int[]index, float val)
	{
		int ind=shapeToIndex(index);
		data.items[ind] = val;
	}
	public void set(Base d)
	{
		int[] tmpSh=new int[shape.length];
		for (int i=0;i < length;i++)
		{
			Util.indexToShape(i, shape, tmpSh);
			set(tmpSh, d.get(tmpSh));
		}
	}
	public void setGrad(Base d)
	{
		int[] tmpSh=new int[shape.length];
		for (int i=0;i < length;i++)
		{
			Util.indexToShape(i, shape, tmpSh);
			setGrad(tmpSh, d.get(tmpSh));
		}
	}
	/*
	 // set multiple values(array value into the data
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
			data.items[i] = values[pos++];
	}
	// this method is the same as set Array but it repeates the same value upto the count range.
	public void set(int[]index, float val, int count)
	{
		int ind=shapeToIndex(index);
		if (count == -1)
			count = data.length - ind;
		for (int i=ind;i < ind + Math.min(count, data.length - ind);i++)
		{
			data.items[i] = val;
		}
	}
	// not tested.
	public float getRaw(int index)
	{
		return data.items[index];
	}
	public float[] getRaw(int index, int count)
	{
		float[] f=new float[count];
		for (int i=0;i < count;i++)
			f[i] = data.items[index + i];
		return f;
	}
	public float[] getRaw(int index, float[]out, int count)
	{
		float[] f=new float[count];
		for (int i=0;i < count;i++)
			f[i] = data.items[index + i];
		return f;
	}
	// not tested.
	public void setRaw(int index, float val)
	{
		data.items[index] = val;
	}
	// not tested.
	public void setRaw(int index, float[] dt)
	{
		int pos=0;
		for (int i=index;i < Math.min(dt.length, data.length - index);i++)
			data.items[index] = dt[pos++];
	}
	public Base reshape(int...newShape)
	{
		fillShape(newShape);
		if (isTransposed())
		{
			Base d=copy().reshapeLocal(newShape);
			return d;
		}
		if (length != length(newShape))
			throw new RuntimeException("can't view this array into " + Arrays.toString(newShape) + " shape. Reason!! the length is not equal");
		Base d=new Base(data, newShape, offset);
		// d.setRequiresGradient(requiresGradient);
		if (d.requiresGradient())
		{
			d.setGradientFunction(reshapeGradient, this);
			// d.setGradientFunction(gradientFunction);
			// d.setGradientParams(params);
			// d.gradient = gradient;
		}
		return d;
	}
	public Base reshapeLocal(int...newShape)
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
	public Base transpose()
	{
		int[] ax=new int[shape.length];
		for (int i=0;i < ax.length;i++)
			ax[i] = ax.length - i - 1;
		return transpose(ax);
	}
	public Base transpose(int...axes)
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
		Base d = new Base(data, sh, strd, offset);
		// d.setRequiresGradient(requiresGradient());
		if (d.requiresGradient())
		{
			d.setGradientFunction(transposeGradient, this);
			// d.setGradientParams(params);
			// d.gradient = gradient;
		}
		return d;
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
	public Base copy()
	{
		if (isTransposed()) // or broadcasted.
		{
			float[] dt=new float[length];
			Base d=new Base(dt, shape).setRequiresGradient(data.requiresGradient);
			if (d.requiresGradient())
				d.setGradientFunction(copyGradient, this);
			// d.setGradientParams(params);
			for (int i=0;i < length;i++)
			{
				int[] shp=indexToShape(i);
				dt[i] = get(shp);
				if (d.requiresGradient())
					d.data.gradient[i] = getGrad(shp);
			}
			return d;
		}
		else
		{
			Base d = new Base(Arrays.copyOf(data.items, data.length), shape);
			d.setRequiresGradient(requiresGradient());
			if (d.requiresGradient())
			{
				d.setGradientFunction(copyGradient, this);
				// d.setGradientParams(params);
				d.data.gradient = Arrays.copyOf(data.gradient, data.length);
			}
			return d;
		}
	}
	// not tested
	// needs some improvements.
	// it must support broadcasting.
	// the new shape can be a broadcast if the original shape.
	// the reshape method can't give that ability.
	public Base copyTo(int...newShape)
	{
		if (Arrays.equals(newShape, shape))
			return copy();
		if (isBrodcastable(newShape))
		{
			int len=length(newShape);
			float[] dt=new float[len];
			Base d=new Base(dt, newShape).setRequiresGradient(requiresGradient());
			if (d.requiresGradient())
				d.setGradientFunction(copyToGradient, this);
			d.setGradientParams(params);
			for (int i=0;i < len;i++)
			{
				int[] shp=indexToShape(i);
				dt[i] = get(shp);
				if (d.requiresGradient())
					d.data.gradient[i] = getGrad(shp);
			}
			return d;
		}
		return copy().reshapeLocal(newShape);
	}
	@Override
	public String toString()
	{
		String inf="shape : " + Arrays.toString(shape) + " len :" + length + (isTransposed() ?" : transposed.": ".") + (requiresGradient() ?" :: gradient = " + requiresGradient(): "");
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
	public Base as1DArray()
	{
		return reshape(-1);
	}
	public Base as2DArray()
	{
		return reshape(-1, shape[shape.length - 1]);
	}
	public Base as3DArray()
	{
		if (shape.length == 1)
		{
			return reshape(-1, 1, shape[shape.length - 1]);
		}
		return reshape(-1, shape[shape.length - 2], shape[shape.length - 1]);
	}
	public Base trim()
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
		if (c == shape.length)
		{
			c = shape.length == 1 ?1: shape.length - 1;
		}
		int[] sh=Arrays.copyOfRange(shape, c, shape.length);
		Base d = new Base(data, sh, strides, offset);
		d.setRequiresGradient(requiresGradient());
		if (d.requiresGradient())
		{
			d.setGradientFunction(trimGradient, this);
			// d.setGradientParams(params);
			// d.data.gradient = data.gradient;
		}
		return d;
	}
}

