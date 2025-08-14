import gss.*;
import gss.arr.*;
import java.util.*;

import static gss.Util.*;

public class TestValue
{
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
				Value sm=new Value(0);	
				for (int c=0;c < a.shape[1];c++)
				{
					sm = sm.add(a.getValue(ar, c).mul(b.getValue(br, c)));
				}
				bs.setValue(sm, Util.ar(ar, br));
			}
		bs.setRequiresGradient(a.requiresGradient() | b.requiresGradient());
		bs = bs.setGradientFunction(GradFunc.dotGradient, a, b).reshape(out);
		return bs;
	}
	public static Base convolve1d2(Base a, Base b, Base out)
	{
		// valid convolution.
		if (a.getDim() != 1 || b.getDim() != 1)
			throw new RuntimeException("convolve 1d error : expected 1d kernel array and 1d data.");
		int len=Math.max(a.length, b.length) - Math.min(a.length, b.length) + 1;
		if (out == null)
			out = new Base(len).setRequiresGradient(a.requiresGradient() | b.requiresGradient());
		int wi=0;
		int iinc=a.length > b.length ?1: 0;
		int kinc=b.length > a.length ?1: 0;
		int wr=b.length > a.length ?len - 1: 0;
		for (int w=0;w < len;w++)
		{
			Value sm=new Value(0);
			int kl=b.length - 1;
			for (int i=0;i < Math.min(a.length, b.length);i++)
			{
				Value aa=b.getValue(kl - wr);
				Value bb=a.getValue(i + wi);
				sm = sm.add(aa.mul(bb));
				kl--;
			}
			wr -= kinc;
			wi += iinc;
			out.setValue(sm, w);
		}
		return out.setGradientFunction(GradFunc.itemGradient);
	}
}
