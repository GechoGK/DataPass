package gss.arr;

import java.util.*;

import static gss.Util.*;

public class Data
{
	public float[] items;
	public float[] gradient;
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
		this.length=items.length;
		this.gradient = grad;
		this.requiresGradient = true;
	}
	public void setGradientEnabled(boolean enable)
	{
		this.requiresGradient = enable;
		if (enable && gradient == null)
			gradient = new float[items.length];
		else
			gradient = null;
	}
	public void zeroGradient()
	{
		if (requiresGradient)
			Arrays.fill(gradient, 0);
	}
}
