package gss.arr;

import gss.*;
import java.io.*;
import java.util.*;

import static gss.Util.*;
import static gss.arr.GradFunc.*;

/*
 --- create a container for
 >> unmodified data.
 >> transposed data.
 >> sliced data -> if it have !=1 jumps
 -- remake all method with the tag @CONT on all containers.
 // that will increase a little performance.
 */

// Basic unmodified container.
public class Base implements Serializable
{
	public static final long serialVersionUID=297474738l;

	public int[] shape;
	public int[] strides;
	public int length;
	public int offset=0;
	// internal data.
	public Data data;

	public GradFunc gradientFunction;
	public List<Base> childs;
	public Object params;

	private int getCount=0;
	public boolean debugGet=false;

	public Base(int...shp)
	{
		if (shp.length == 0)
			error("");
		this.shape = Util.copy(shp);
		this.strides = genStrides(shape);
		this.length = length(shape);
		this.data = new Data(shape);
		this.offset = 0;
	}
	public Base(float[] src)
	{
		this.data = new Data(src);
		this.shape = new int[]{data.length};
		this.strides = new int[]{1};
		this.length = length(shape);
		this.offset = 0;
	}
	public Base(float[] src, int...shp)
	{
		this.data = new Data(src);
		this.shape = Util.copy(shp);
		this.strides = genStrides(shape);
		this.length = length(shape);
		this.offset = 0;
		if (length != data.length)
			throw new RuntimeException("invalid shape or data. they are not equal in length.");
	}
//	public Base(float[] src, int[]shp, int[] strd, int off)
//	{
////		this.data = new Data(src);
////		this.shape = shp;
////		this.strides = strd;
////		this.offset = off;
////		this.length = length(shape);
//		newBase(new Data(src), shp, strd, off);
//	}
//	public Base(Data d, int[]shp, int off)
//	{
////		this.data = d;
////		this.shape = shp;
////		this.strides = genStrides(shape);
////		this.length = length(shape);
////		this.offset = off;
//		newBase(d, shp, genStrides(shp), off);
//	}
//	public Base(Data d, int[]shp, int[]strd, int off)
//	{
////		this.data = d;
////		this.shape = shp;
////		this.strides = strd;
////		this.length = length(shape);
////		this.offset = off;
//		newBase(d, shp, strd, off);
//	}
	protected Base newBase(int[]shp)
	{
		return new Base(shp);
	}
	protected Base newBase(float[]src)
	{
		return new Base(src);
	}
	protected Base newBase(float[]src, int...shp)
	{
		if (length(shp) != src.length)
			error("the length of the array must be equal to the shape total length");
		Base b=newBase(src);
		b.shape = Util.copy(shp);
		b.strides = genStrides(shp);
		b.length = length(shp);
		b.offset = 0;
		return b;
	}
	public Base newBase(int[]shp, int[]strd, int off)
	{
		Base b=newBase(shp);
		b.strides = Util.copy(strd);
		b.length = length(shp);
		b.offset = off;
		return b;
	}
	public Base newBase(Data d, int[]shp, int off)
	{
		Base b=newBase(shp);
		b.data = d;
		b.length = length(shp);
		b.offset = off;
		return b;
	}
	public Base newBase(Data d, int[]shp, int[]strd, int off)
	{
		Base b=newBase(shp);
		b.data = d;
		b.strides = Util.copy(strd);
		b.length = length(shp);
		b.offset = off;
		return b;
	}
	public int getDim()
	{
		return shape.length;
	}
	public Base slice(int...ind)
	{
		// lazy slicing
		// doesn't check index length.
		if (ind.length > shape.length)
			throw new RuntimeException("index.length must not be greater than shape.length:(" + Arrays.toString(ind) + ", " + Arrays.toString(shape) + ")");
		int off=shapeToIndex(Arrays.copyOf(ind, shape.length));
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
		Base b = newBase(data, sh, str, off);
		b.setGradientFunctionS(stepGradient, this);
		return b;
	}
	public Base slice(int[]...ind)
	{
		// lazy slicing
		// doesn't check the length of array.
		// need some improvement.
		// if ind length not equal to shape length.
		// it must fill the rest by default.  fix it.
		// the ind array expexted to be 3 inength(start, end, increment);
		if (ind.length > shape.length)
			throw new RuntimeException("index.length must not be greater than shape.length.");
		// int shl=shape.length - ind.length;
		// if (shl == 0)
		//	throw new RuntimeException("invalid index for slice.");
		// System.out.println("slicing with range :");
		int[] sh = new int[shape.length];
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
			// check if ln > shape[i]; then throw out of range error.
			if (ln > shape[i])
				error("index out of range" + Arrays.toString(r) + "reason :" + r[1] + " > " + shape[i]);
			sh[i] = ln;
			str[i] = strides[i] * r[2];
			off += r[0] * strides[i];
			i++;
		}
		// System.out.println("offset = " + off + ", new shape :" + Arrays.toString(sh) + ", new stride :" + Arrays.toString(str));
		Base b = newBase(data, sh, str, off);
		b.setGradientFunctionS(stepGradient, this);
		return b;
	}
	// @CONT
	public void fill(float v)
	{
		// don't fill all items.
		// instead loop over items and set their value.
		if (!isOriginal())
		{
			// print("off :" + offset + ", " + length);
			int[]tmpSh=new int[shape.length];
			for (int i=0;i < offset + length;i++)
			{
				tmpSh = indexToShape(i);
				set(tmpSh, v);
			}
		}
		else
			Arrays.fill(data.items, offset, offset + length, v);
	}
	// @CONT
	public int shapeToIndex(int...index)
	{
		int newPos=0;
		int strShPos = Math.max(0, shape.length - index.length);
		int strIndPos = Math.max(0, index.length - shape.length);
		for (int i = 0;i < Math.min(shape.length, index.length);i++)
		{
			int shps=i + strShPos;
			int indps=i + strIndPos;
			// strict broadcasting.
			// to enable, uncomment the if code block below.
//			if (Mode.isStrictBroadcastEnabled() && shape[shps] != 1 && index[indps] >= shape[shps]){
//				error("index out of bound exception " + index[indps] + " >= " + shape[shps]);
//			}
			int shapeInd = Math.min(index[indps], shape[shps] - 1);
			// lazy broadcasting enabled --^
			// to disable use
			// int shapeInd = index[i+strIndPos] // it will automatically throw error when out of range.
			newPos += shapeInd *  strides[shps];
		}
		return newPos + offset;
	}
	// @CONT
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
	// @CONT...
	public float get(int...index)
	{
		int ind=Math.max(0, shapeToIndex(index));
		float dt= data.items[ind];
		if (debugGet)
			System.out.println(getCount++ + ". get at(" + Arrays.toString(index) + ") = " + dt);
		return dt;
	}
	// @CONT
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
	// @CONT
	// set single value to the data.
	public void append(int[]index, float val)
	{
		int ind=shapeToIndex(index);
		data.items[ind] += val;
	}
	// @CONT
	public void set(int[]index, float val)
	{
		int ind=shapeToIndex(index);
		data.items[ind] = val;
	}
	// @CONT
	public void set(Base d)
	{
		// lazy set value.
		// it doesn't check the dimension equality and the length equality.
		if (d.shape.length != this.shape.length) // dimension is not equal
			throw new RuntimeException("invalid dimension to set.");
		int[] tmpSh=new int[shape.length];
		for (int i=0;i < length;i++)
		{
			Util.indexToShape(i, shape, tmpSh);
			set(tmpSh, d.get(tmpSh));
		}
	}
	// same as set(Base d) but it doesn't overwrite values.
	// @CONT
	public void append(Base d)
	{
		// lazy set value.
		// it doesn't check the dimension equality and the length equality.
		if (d.shape.length != this.shape.length) // dimension is not equal
			throw new RuntimeException("invalid dimension to set.");
		int[] tmpSh=new int[shape.length];
		for (int i=0;i < length;i++)
		{
			Util.indexToShape(i, shape, tmpSh);
			append(tmpSh, d.get(tmpSh));
		}
	}
	/*
	 // set multiple values(array value into the data
	 // @startIndex[] which start index of the destination data.
	 // @values[] float value that is going to copied.
	 !! if the values array is greater than the destination array,
	 // it will write up to the maximum data it can hold, the rest will trimmed.

	 */
	// !! does it work on transposed array ?????
//	public void set(int[]startIndex, float...values)
//	{
//		int ind=shapeToIndex(startIndex);
//		int pos=0;
//		for (int i=ind;i < ind + Math.min(values.length, data.length - ind);i++)
//			data.items[i] = values[pos++];
//	}
//	// this method is the same as set Array but it repeates the same value upto the count range.
//	public void set(int[]index, float val, int count)
//	{
//		int ind=shapeToIndex(index);
//		if (count == -1)
//			count = data.length - ind;
//		for (int i=ind;i < ind + Math.min(count, data.length - ind);i++)
//		{
//			data.items[i] = val;
//		}
//	}
	// not tested.
	public float getRaw(int index)
	{
		if (index >= data.length && index < offset)
			throw new IndexOutOfBoundsException("invalid index " + index + " it must be less than (" + length + ")");
		return data.items[offset + index];
	}
//	// not tested.
//	public float[] getRaw(int index, int count)
//	{
//		float[] f=new float[count];
//		for (int i=0;i < count;i++)
//			f[i] = data.items[index + i];
//		return f;
//	}
//	// not tested.
//	// out array size must be greater than count.
//	public float[] getRaw(int index, float[]out, int count)
//	{
//		if (out.length < count)
//			throw new RuntimeException("give an array that can hold up to count @count :" + count);
//		for (int i=0;i < count;i++)
//			out[i] = data.items[index + i];
//		return out;
//	}
	// not tested.
	public void setRaw(int index, float val)
	{
		if (index >= length && index < offset)
			throw new IndexOutOfBoundsException("invalid index " + index + " it must be less than (" + length + ")");
		data.items[offset + index] = val;
	}
	// not tested.
//	public void setRaw(int index, float[] dt)
//	{
//		int pos=0;
//		for (int i=index;i < Math.min(dt.length, data.length - index);i++)
//			data.items[index] = dt[pos++];
//	}
	// @CONT
	// data manipulation functions.
	public Base reshape(int...newShape)
	{
		fillShape(newShape);
		if (isTransposed())
		{
			// print("reshaping transposed");
			Base d=copy().reshape(newShape);
			// print(d.gradientFunction);
			return d;
		}
		// print(length, newShape);
		if (length != length(newShape))
			throw new RuntimeException("can't view this array into " + Arrays.toString(newShape) + " shape. Reason!! the length is not equal");
		Base d=newBase(data, newShape, offset);
		d.setRequiresGradient(hasGradient());
		d.setGradientFunctionS(stepGradient, this);
		return d;
	}
	public Base reshapeLocal(int...newShape)
	{
		fillShape(newShape);
		if (length != length(newShape))
			error("unable to reshape the array to " + Arrays.toString(newShape) + ", array length of array doesn't match to shape length. array length(" + length + ") != shape length(" + length(newShape) + ")");
		if (isTransposed())
			error("unable to modify the shape of transposed array.");
		this.shape = Util.copy(newShape);
		this.strides = genStrides(shape);
		this.length = length(shape);
		// error in sub array --v uncomment to see it.
		// if (length != data.length)
		// 	throw new RuntimeException("invalid shape or data. they are not equal in length.");
		return this;
	}
	// @CONT
	public Base transpose()
	{
		int[] ax=new int[shape.length];
		for (int i=0;i < ax.length;i++)
			ax[i] = ax.length - i - 1;
		return transpose(ax);
	}
	// @CONT
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
		Base d = newBase(data, sh, strd, offset);
		d.setRequiresGradient(hasGradient());
		if (d.hasGradient())
		{
			d.setGradientFunction(stepGradient, this);
//			// d.setGradientParams(params);
//			// d.gradient = gradient;
		}
		return d;
	}
	private int[] fillShape(int...shp)
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
	public boolean isSliced()
	{
		return data.length != length;
	}
	public boolean isOriginal()
	{
		return !isTransposed() && !isSliced();
	}
	// @CONT
	public Base copy()
	{
		// System.out.println("copying...");
		if (!isOriginal()) // or broadcasted.
		{
			// print("copy transposed");
			float[] dt=new float[length];
			Base d=newBase(dt, shape);
			d.setRequiresGradient(hasGradient());
			if (d.hasGradient())
				d.setGradientFunction(copyGradient, this);
			// print(d.gradientFunction);
			// d.setGradientParams(params);
			for (int i=0;i < length;i++)
			{
				int[] shp=indexToShape(i);
				dt[i] = get(shp);
				if (d.hasGradient())
					d.setRawGrad(i,  getGrad(shp));
			}
			return d;
		}
		else
		{
			// print("copy non-transposed");
			// don't use data.items.
			Base d = newBase(Arrays.copyOfRange(data.items, offset, offset + length), shape);
			d.setRequiresGradient(hasGradient());
			if (d.hasGradient())
			{
				d.setGradientFunction(copyGradient, this);
//				// d.setGradientParams(params);
				d.data.gradient = Arrays.copyOfRange(data.gradient, offset, offset + data.length); // wrong.
			}
			return d;
		}
	}
	// @CONT
	public Base deepCopy()
	{
		// System.out.println("copying...");
		if (!isOriginal()) // or broadcasted.
		{
			// print("copy transposed");
			float[] dt=new float[length];
			Base d=newBase(dt, shape);
			d.setRequiresGradient(hasGradient());
			// print(d.gradientFunction);
			// d.setGradientParams(params);
			for (int i=0;i < length;i++)
			{
				int[] shp=indexToShape(i);
				dt[i] = get(shp);
				if (d.hasGradient())
					d.setRawGrad(i,  getGrad(shp));
			}
			return d;
		}
		else
		{
			// print("copy non-transposed");
			// don't use data.items.
			Base d = newBase(Arrays.copyOfRange(data.items, offset, offset + length), shape);
			d.setRequiresGradient(hasGradient());
			if (d.hasGradient())
			{
				d.data.gradient = Arrays.copyOfRange(data.gradient, offset, offset + data.length);
			}
			return d;
		}
	}
	@Override
	public String toString()
	{
		String inf="shape : " + Arrays.toString(shape) + " len :" + length + (isTransposed() ?" : transposed.": ".") + (hasGradient() ?"haveGradient? : true": "");
		return inf + "\n" + getArrayAsString();
	}
	public void printArray()
	{
		print(toString());
	}
	public String getArrayAsString()
	{
		StringBuilder sb=new StringBuilder();
		genArray(this, sb, 0);
		// printArray2(cShape, 0, sb, "");
		String s=sb.toString().trim();
		if (s.endsWith(","))
			s = s.substring(0, s.length() - 1);
		return s;
	}
	private void genArray(Base b, StringBuilder sb, int idt)
	{
		String indent=Util.getString(" ", idt);
		if (b.getDim() == 1)
		{
			sb.append(indent);
			sb.append("[");
			for (int i=0;i < b.shape[0];i++)
			{
				if (i != 0)
					sb.append(", ");
				sb.append(b.get(i));
			}
			sb.append("],\n");
			return;
		}
		sb.append(indent);
		sb.append("[\n");
		for (int i=0;i < b.shape[0];i++)
			genArray(b.slice(i), sb, idt + 2);
		sb.append(indent);
		sb.append("],\n");
	}
	public Base as1DArray()
	{
		if (getDim() == 1)
			return this;
		return reshape(-1);
	}
	public Base as2DArray()
	{
		if (getDim() == 2)
			return this;
		return reshape(-1, shape[shape.length - 1]);
	}
	public Base as3DArray()
	{
		if (getDim() == 3)
			return this;
		if (shape.length == 1)
		{
			return reshape(-1, 1, shape[shape.length - 1]);
		}
		return reshape(-1, shape[shape.length - 2], shape[shape.length - 1]);
	}
	public Base as4DArray()
	{
		if (getDim() == 4)
			return this;
		if (shape.length == 1)
		{
			return reshape(-1, 1, 1, shape[shape.length - 1]);
		}
		else if (shape.length == 2)
		{
			return reshape(-1, 1, shape[shape.length - 2], shape[shape.length - 1]);
		}
		return reshape(-1, shape[shape.length - 3], shape[shape.length - 2], shape[shape.length - 1]);
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
		int[] strd=Arrays.copyOfRange(strides, c, strides.length);
		Base d = newBase(data, sh, strd, offset);
		d.setRequiresGradient(hasGradient());
		if (d.hasGradient())
		{
			d.setGradientFunction(stepGradient, this);
			// d.setGradientParams(params);
			// d.data.gradient = data.gradient;
		}
		return d;
	}
	// @CONT
	// set and get method for specific dim.
	public void set1d(int x, float val)
	{
		if (isOriginal())
			setRaw(x, val);
		else if (getDim() == 1)
			set(ar(x), val); 
		else
			set(indexToShape(x), val);
	}
	// @CONT
	public float get1d(int x)
	{
		if (isOriginal())
			return getRaw(x);
		else if (shape.length == 1)
		 	return get(ar(x));
		else
			return get(indexToShape(x));
	}
////// gradient section.
	public boolean hasGradient()
	{
		return data.hasGradient;
	}
	public Base setRequiresGradient(boolean b)
	{
		data.setGradientEnabled(b);
		return this;
	}
	public Base detachGradient()
	{
		if (!hasGradient())
			error("the array has no gradient to detach. please enable first by calling arr.setRequiresGradient(true);");
		return newBase(new Data(data.gradient), shape, strides, offset);
	}
////// /*
	public void setGrad(float v)
	{
		fillGrad(v);
	}
	// @CONT
	// --v  this function appends the gradient value.
	public void setGrad(int[]index, float val)
	{
		int ind=shapeToIndex(index);
		data.gradient[ind] += val;
	}
	// @CONT
	// --v this function overwrite the value of gradient into a new one, used for zero gradient.
	public void setGrad2(int[]index, float val)
	{
		int ind=shapeToIndex(index);
		data.gradient[ind] = val;
	}
	// @CONT
	public float getGrad(int...index)
	{
		int ind=Math.max(0, shapeToIndex(index));
		return data.gradient[ind];
	}
	// @CONT
	public void fillGrad(float v)
	{
		if (!hasGradient())
			error("the array has no gradient to fill. please enable first by calling arr.setRequiresGradient(true);");
		// don't fill using Array.fill.
		// instead loop over items and set their value.
		if (!isOriginal())
		{
			// print("off :" + offset + ", " + length);
			int[]tmpSh=new int[shape.length];
			for (int i=0;i < offset + length;i++)
			{
				tmpSh = indexToShape(i);
				setGrad(tmpSh, v);
			}
		}
		else
		{
			for (int i=0;i < length;i++)
				setRawGrad(i, v);
			// Arrays.fill(data.gradient, offset, offset + length, v);
		}
	}
	// @CONT
	public void setGrad(Base d)
	{
		if (!hasGradient())
			error("the array has no gradient to set. please enable first by calling arr.setRequiresGradient(true);");
		// lazy set value.
		// it doesn't check the dimension equality and the length equality.
		if (d.shape.length != this.shape.length) // dimension is not equal
			throw new RuntimeException("invalid dimension to set.");
		int[] tmpSh=new int[shape.length];
		for (int i=0;i < length;i++)
		{
			Util.indexToShape(i, shape, tmpSh);
			setGrad(tmpSh, d.get(tmpSh));
		}
	}
	// @CONT
	public void set1dGrad(int x, float val)
	{
		if (isOriginal())
			setRawGrad(x, val);
		else if (getDim() == 1)
			setGrad(ar(x), val); 
		else
			setGrad(indexToShape(x), val);
	}
	// @CONT
	public float get1dGrad(int x)
	{
		if (isOriginal())
			return getRawGrad(x);
		else if (shape.length == 1)
			return getGrad(ar(x));
		else
			return getGrad(indexToShape(x));
	}
	public Base setGradientFunctionS(GradFunc func, Base...childs)
	{
		if (!hasGradient())
			return this;
		return setGradientFunction(func, null, childs);
	}
	public Base setGradientFunctionS(GradFunc func, Object prm, Base...childs)
	{
		if (!hasGradient())
			return this;
		return setGradientFunction(func, prm, childs);
	}
	public Base setGradientFunction(GradFunc func, Base...chlds)
	{
		return setGradientFunction(func, null, chlds);
	}
	public Base setGradientFunction(GradFunc func, Object prms, Base...chlds)
	{
		if (!hasGradient())
			error("gradient not enabled. enable by calling base.setRequiresGradient();");
		if (childs == null)
			childs = new ArrayList<>();
		for (Base b:chlds)
			childs.add(b);
		gradientFunction = func;
		params = prms;
		return this;
	}
	public Base backward()
	{
		// fix it.
		// backward method should be in an new method, which collects all childs and apply backward method.
		// that helps to prevent reccursion functions.
		if (!hasGradient() || gradientFunction == null)
			return this;
		// error("no gradient function found for backward pass.");
		gradientFunction.backward(this, childs.toArray(new Base[0]), params);
		for (Base b:childs)
			if (b != null)
				b.backward();
		return this;
	}
	public Base zeroGrad()
	{
		// fillGrad(0);
		for (int i=0;i < length;i++)
		{
			int[] sh=indexToShape(i);
			setGrad2(sh, 0); // we use setGrad2 to prevent appending value.
		}
		return this;
	}
	public void setRawGrad(int index, float val)
	{
		if (index >= length && index < offset)
			throw new IndexOutOfBoundsException("invalid index " + index + " it must be less than (" + length + ")");
		data.gradient[offset + index] += val;
	}
	public float getRawGrad(int index)
	{
		if (index >= data.length && index < offset)
			throw new IndexOutOfBoundsException("invalid index " + index + " it must be less than (" + length + ")");
		return data.gradient[offset + index];
	}
////// */
	// value area.
	public Base setValue(Value v, int...shp)
	{
		data.setValue(shapeToIndex(shp), v);
		return this;
	}
	public Value getValue(int...shp)
	{
		return data.getValue(shapeToIndex(shp));
	}
}
