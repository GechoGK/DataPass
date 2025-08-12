import gss.arr.*;

import static gss.Util.*;

public class TestValue
{
	public static Base convolve1d2(Base d1, Base kr)
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
	public static Base relu(Base arr)
	{
		// Value[] f=arr.base.toValueArray();
		Base out=new Base(arr.shape).setRequiresGradient(arr.requiresGradient());
		out.setGradientFunction(GradFunc.itemGradient);
		for (int i=0;i < arr.length;i++)
		{
			Value v=arr.getValue(arr.indexToShape(i));
			v = v.getData() > 0 ?v.step(): new Value(0);
			out.setValue(v, out.indexToShape(i));
		}
		return out;
	}
	public static Base sigmoid(Base arr)
	{
		/*
		 float sigmoidForward(float x) {
		 return 1.0f / (1.0f + (float) Math.exp(-x));
		 }
		 */
		// Value[] f=arr.base.toValueArray();
		Base out=new Base(arr.shape).setRequiresGradient(arr.requiresGradient());
		out.setGradientFunction(GradFunc.itemGradient);
		for (int i=0;i < arr.length;i++)
		{
			Value v=arr.getValue(arr.indexToShape(i));
			v = new Value(1).div(new Value(1).add(v.mul(new Value(-1)).exp()));
			out.setValue(v, out.indexToShape(i));
		}
		return out;
	}
	public static Base tanh(Base arr)
	{
		// Value[] f=arr.base.toValueArray();
		Base out=new Base(arr.shape).setRequiresGradient(arr.requiresGradient());
		out.setGradientFunction(GradFunc.itemGradient);
		for (int i=0;i < arr.length;i++)
		{
			Value v=arr.getValue(arr.indexToShape(i)).tanh();
			out.setValue(v, out.indexToShape(i));
		}
		return out;
	}
	public static Base dot2(Base a, Base b)
	{
		int[] out=NDArray.dotShape(a.shape, b.shape);
		a = a.as2DArray();
		if (b.getDim() < 2)
			b = b.reshape(1, -1);
		else if (b.getDim() == 2)
		{
			b = b.transpose(1, 0);
		}
		else if (b.getDim() > 2)
		{
			b = b.transpose(NDArray.dotAxis(b.getDim()));
			b = b.reshape(-1, b.shape[b.shape.length - 1]);
		}
		if (b.shape[b.shape.length - 1] != a.shape[a.shape.length - 1])
			throw new RuntimeException("invalid shape dor dot product.");
		int[] sh={a.shape[0],b.shape[0]};
		Base bs=new Base(sh);
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
		bs.setRequiresGradient(a.requiresGradient() | b.requiresGradient());
		bs = bs.setGradientFunction(GradFunc.dotGradient, a, b).reshape(out);
		return bs;
	}
}
