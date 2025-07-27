package gss.layers;

import gss.*;
import gss.arr.*;

public class Linear extends Module
{
	private boolean hasBiase;
	public Data weight;
	public Data biase;

	public Linear(int in, int out)
	{
		this(in, out, true);
	}
	public Linear(int in, int out, boolean hasBiase)
	{
		this.hasBiase = hasBiase;
		init(in, out);
	}
	private void init(int in, int out)
	{
		weight = newParam(NDArray.rand(in, out)); // .setEnableGradient(true));
		if (hasBiase)
			biase = newParam(NDArray.ones(out)); // .setEnableGradient(true));
	}
	@Override
	public Data forward(Data input)
	{
		Data out =NDArray.dot(input, weight);
		if (hasBiase)
			out = NDArray.add(out, biase);
		return out;
	}
}
