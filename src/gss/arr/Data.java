package gss.arr;

import java.util.*;

import static gss.Util.*;

public class Data
{
	public float[] items;
	public float[] gradient;
	public Value[] gradValues;
	public boolean requiresGradient;
	public int length=0;

	public Data(int...shape)
	{
		items = new float[length(shape)];
		this.length = items.length;
	}
	public Data(int[]shape, boolean enableGrad)
	{
		items = new float[length(shape)];
		this.length = items.length;
		setGradientEnabled(enableGrad);
	}
	public Data(float[] f)
	{
		this.items = f;
		this.length = items.length;
	}
	public Data(float[] f, boolean enableGrad)
	{
		this.items = f;
		this.length = items.length;
		setGradientEnabled(enableGrad);
	}
	public Data(float[] f, float[] grad)
	{
		this.items = f;
		this.length = items.length;
		this.gradient = grad;
		this.requiresGradient = true;
	}
	public void setGradientEnabled(boolean enable)
	{
		this.requiresGradient = enable;
		if (enable)
		{
			if (gradient == null)
				gradient = new float[items.length];
			if (gradValues == null)
				gradValues = new DValue[items.length];
		}
		else
		{
			gradient = null;
			gradValues = null;
		}
	}
	public void zeroGradient()
	{
		if (requiresGradient)
			Arrays.fill(gradient, 0);
	}
	public Value getValue(int pos)
	{
		if (gradValues == null)
			gradValues = new DValue[items.length];
		Value v=gradValues[pos];
		if (v == null)	
		{
			v = new DValue(this, pos);
			gradValues[pos] = v;
		}
		return v;
	}
	public Value setValue(int pos, Value v)
	{
		// System.out.println("setting flat " + ind + " = " + v);
		if (gradValues == null)
		 	gradValues = new DValue[items.length];
		DValue dv=(DValue)gradValues[pos];
		if (dv == null)
		{
			dv = new DValue(this, pos);
			gradValues[pos] = dv;
		}
		dv.set(v);
		// System.out.println(dv);
		return dv;
	}
}
