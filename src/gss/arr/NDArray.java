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
	/*
	 TO-DO
	 -- mode.
	 -- median.
	 >> export.
	 >> import.
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
	public static Base sum(final Base b, final int axis)
	{
		int[] sumShape=removeAtIndex(b.shape, axis);
		final Base out=empty(sumShape).setRequiresGradient(b.hasGradient());
		final int count=b.shape[axis];
		loop(sumShape, new ArrayToFloatFunction(){
				@Override
				public float apply(int[] p1)
				{
					float sm=0;
					for (int i=0;i < count;i++)
					{
						int[]sh=putAtIndex(p1, i, axis);
						sm += b.get(sh);
					}
					out.set(p1, sm);
					return 0;
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
	// import and export arrays
	public static void save(Base ar, String path)
	{
		saveJSON(ar, path);
	}
	public static void save(Base ar, String path, FileType type)
	{
		if (type == FileType.JSON)
			saveJSON(ar, path);
		else if (type == FileType.TEXT)
			saveTEXT(ar, path);
		else if (type == FileType.BINARY)
			saveBINARY(ar, path);
		else throw new RuntimeException("unknown File type (valid files are (JSON, TEXT, BINARY)) instead found ::" + type);
	}
	public static void saveJSON(Base ar, String path)
	{
		try
		{
			// copy the array to prevent missing values because of strides and offsets.
			ar = ar.copy();
			// shape is the same.
			// stride will be generated from shape, not copied.
			// offset will be 0.
			/*
			 protocol...
			 type = float -> string
			 requireGradient = false -> boolean
			 shape = [2,3] -> array
			 array = [1,2,3,...] -> array

			 */
			JSONObject obj=toJson(ar);
			String jsonString=obj.toString(3);
			Util.saveString(path, jsonString);
			// Util.print("array saved as json file");
		}
		catch (Exception e)
		{
			Util.print("error :" + e);
			e.printStackTrace();
		}
	}
	public static JSONObject toJson(Base ar) throws Exception
	{
		JSONObject obj=new JSONObject("{}");
		obj.put("type", "float");
		obj.put("requiresGradient", ar.hasGradient());
		JSONArray arr=new JSONArray();
		int[]sh=ar.shape;
		for (int s:sh)
			arr.put(s);
		obj.put("shape", arr);
		float[]dt=ar.data.items;
		arr = new JSONArray();
		for (float f:dt)
			arr.put(f);
		obj.put("array", arr);
		return obj;
	}
	public static void saveTEXT(Base ar, String path)
	{
		/*
		 protocols
		 type = float
		 requiresGradient = false
		 shape = [1,2,3]
		 array = [1,2,3,...]
		 */
		try
		{
			// copy the array to prevent missing values because of strides and offsets.
			ar = ar.copy();
			// shape is the same.
			// stride will be generated from shape, not copied.
			// offset will be 0.
			StringBuilder sb=new StringBuilder();
			sb.append("type = float\n");
			sb.append("requiresGradient = ");
			sb.append(ar.hasGradient());
			sb.append("\n");
			sb.append("shape = ");
			sb.append(Arrays.toString(ar.shape).replace(" ", ""));
			sb.append("\n");
			sb.append("array = ");
			sb.append(Arrays.toString(ar.data.items).replace(" ", ""));
			sb.append("\n");
			String textData=sb.toString();
			Util.saveString(path, textData);
			// Util.print("array saved as text file");
		}
		catch (Exception e)
		{
			Util.print("error :" + e);
			e.printStackTrace();
		}
	}
	public static void saveBINARY(Base ar, String path)
	{
		try
		{
			// copy the array to prevent missing values because of strides and offsets.
			ar = ar.copy();
			// shape is the same.
			// stride will be generated from shape, not copied.
			// offset will be 0.
			/*
			 binary format
			 —————————————
			 |  1  | 0|1 | ->type : 1 = float | requiresGrad : 0|1 false or true
			 —————————————
			 .. ^--- int
			 int -> length of shape.
			 -- 1,2,3.4,5.6,7...--
			 .. ^--- int
			 int -> length of data.
			 -- 1.0, 1.3, 4.8... --
			 .. ^--- float
			 */
			DataOutputStream dos=new DataOutputStream(new FileOutputStream(path));
			int tpgrd=0b10000000;
			if (ar.hasGradient())
				tpgrd |= 0b01000000; // requiresGrad true
			else
				tpgrd |= 0b00000000; // requoresGrad false.
			dos.writeInt(tpgrd); // write type and requiresGradient -> int
			int[] sh=ar.shape;
			dos.writeByte(sh.length); // write shape length -> byte
			for (int s:sh)
				dos.writeInt(s); // write each individual shape items. -> int
			float[] dt=ar.data.items;
			dos.writeInt(dt.length); // write array data -> int
			for (float f:dt)
				dos.writeFloat(f); // write each individual array items. -> float
			dos.flush();
			dos.close(); // done saving.
			// Util.print("array saved as binary file");
		}
		catch (Exception e)
		{
			Util.print("error :" + e);
			e.printStackTrace();
		}
	}
	public static Base load(String path) throws IOException,JSONException
	{
		/*
		 // the type of file is determined poorly by the extension of the file.
		 .txt = textFile (FileType.TEXT)
		 .json = jsonFile (FileType.JSON)
		 .bin = binaryFile (FileType.BINARY)
		 */
		String ext=path.substring(path.lastIndexOf(".")).toLowerCase();
		if (ext.equals(".txt"))
			return loadText(path);
		else if (ext.equals(".json"))
			return loadJSON(path);
		else if (ext.equals(".ndbin"))
			return loadBinary(path);
		else throw new RuntimeException("unknown file type with extension :" + ext + ",  or pass FileType to determine the typeFile. load(String,FileType);"); 
	}
	public static Base load(String path, FileType type) throws IOException,JSONException
	{
		if (type == FileType.JSON)
			return loadJSON(path);
		else if (type == FileType.TEXT)
			return loadText(path);
		else if (type == FileType.BINARY)
			return loadBinary(path);
		else throw new RuntimeException("unknown FileType :" + type);
	}
	public static Base loadJSON(String path) throws JSONException,IOException
	{
		/*
		 protocol...
		 type = float -> string
		 requireGradient = false -> boolean
		 shape = [2,3] -> array
		 array = [1,2,3,...] -> array

		 */
		String jsonText=Util.readString(path);
		JSONObject obj=new JSONObject(jsonText);
		// ignore type for now.
		boolean grad=obj.getBoolean("requiresGradient"); // get requiresGrad -> string
		JSONArray arr=obj.getJSONArray("shape"); // getShape ---v parse it. -> int
		int[]shape=new int[arr.length()];
		for (int i=0;i < shape.length;i++)
			shape[i] = arr.getInt(i);
		arr = obj.getJSONArray("array"); // get the array data. ---v parse it. -> float
		float[] arrayData=new float[arr.length()];
		for (int i=0;i < arrayData.length;i++)
			arrayData[i] = (float)arr.getDouble(i);
		// done prepare the NDArray.
		Base arOut=NDArray.wrap(arrayData, shape).setRequiresGradient(grad);
		return arOut;
	}
	public static Base loadText(String path) throws IOException
	{
		/*
		 protocols
		 type = float
		 requiresGradient = false
		 shape = [1,2,3]
		 array = [1,2,3,...]
		 */
		String txt=Util.readString(path);
		// split the text by new line.
		String[] lines=txt.split("\n");
		// iterate over each line and parse the result.
		boolean grad=false;
		int[] shape=null;
		float[] arrData=null;
		for (String line:lines)
		{
			// each line is key and value separated using "=", if the line doesn't contain "=" oass to next.
			if (!line.contains("="))
				continue;
			String key=line.substring(0, line.indexOf("=")).trim();
			String val= line.substring(line.indexOf("=") + 1).trim();
			// Util.print("key =" + key + "||| value =" + val);
			if (key.equals("type"))
			{/*ignored for now*/}
			else if (key.equals("requiresGradient"))
			{
				// parse value as boolean.
				grad = Boolean.valueOf(val);
			}
			else if (key.equals("shape"))
			{
				// the type of shape us array, we need to parde it.
				val = val.substring(1, val.length() - 1).trim(); // trying to remove "[" and "]"
				String[] shArrs=val.split(",");
				shape = new int[shArrs.length];
				for (int i=0;i < shArrs.length;i++)
				{
					if (shArrs[i].trim().length() == 0) // sometimes empty "," may found. so skip it.
						continue;
					shape[i] = Integer.parseInt(shArrs[i]);
				}
			}
			else if (key.equals("array")) // the same as shape, copied!
			{
				// the type of shape us array, we need to parde it.
				val = val.substring(1, val.length() - 1).trim(); // trying to remove "[" and "]"
				String[] shArrs=val.split(",");
				arrData = new float[shArrs.length];
				for (int i=0;i < shArrs.length;i++)
				{
					if (shArrs[i].trim().length() == 0) // sometimes empty "," may found. so skip it.
						continue;
					arrData[i] = Float.parseFloat(shArrs[i]);
				}
			}
		}
		if (shape == null) // if shape is null there is no way to construct the array. so throw an error.
			throw new RuntimeException("unable to read shape!!!");
		// finally prepare the array and return it.
		Base arrOut=null;
		if (arrData == null)// if arrayData is null we can construct the array with "0"s inside, bu5 inform the user that arrayData can't be read.
		{
			Util.print("unable to read array data returning with array filled with \"0\"s");
			arrOut = NDArray.zeros(shape).setRequiresGradient(grad);
		}
		else
			arrOut = NDArray.wrap(arrData, shape).setRequiresGradient(grad);
		return arrOut;
	}
	public static Base loadBinary(String path) throws IOException
	{
		/*
		 binary format
		 —————————————
		 |  1  | 0|1 | ->type : 1 = float | requiresGrad : 0|1 false or true
		 —————————————
		 .. ^--- int
		 int -> length of shape.
		 -- 1,2,3.4,5.6,7...--
		 .. ^--- int
		 int -> length of data.
		 -- 1.0, 1.3, 4.8... --
		 .. ^--- float
		 */
		DataInputStream dis=new DataInputStream(new FileInputStream(path));
		boolean grad=false;
		int[] shape=null;
		float[] arrData=null;
		// read type but ignored.
		int flag=dis.readInt();
		int type=flag & 0b10000000; // type 0b10000000 = float 0b00000000 = int. atleast for now. @not used.
		//   ^--- type no use here.
		int gradFlag=flag & 0b01000000; // grad 0b01000000 = true 0b00000000 = false;
		grad = gradFlag == 0b01000000 ?true: false;
		byte shapeLen=dis.readByte(); // length of the shape;
		shape = new int[shapeLen];
		for (int i=0;i < shapeLen;i++)
			shape[i] = dis.readInt();
		// next we read array data length.
		int dataLen=dis.readInt();
		arrData = new float[dataLen];
		for (int i=0;i < dataLen;i++)
			arrData[i] = dis.readFloat();
		// done prepare array.
		if (shape == null) // if shape is null there is no way to construct the array. so throw an error.
			throw new RuntimeException("unabke to read shape");
		Base arrOut=null;
		if (arrData == null) // if arrayData is null we can construct the array with "0"s inside, bu5 inform the user that arrayData can't be read.
		{
			Util.print("unable to read array data returning with array filled with \"0\"s");
			arrOut = NDArray.zeros(shape).setRequiresGradient(grad);
		}
		else
			arrOut = NDArray.wrap(arrData, shape).setRequiresGradient(grad);
		return arrOut;
	}
	public enum FileType
	{
		JSON,
		TEXT,
		BINARY;
	}
	public static void saveModule(Module m, String path) throws Exception
	{
		Util.saveObject(path, m);
	}
	public static Module loadModule(String path) throws Exception
	{
		return (Module)Util.loadObject(path);
	}
}
