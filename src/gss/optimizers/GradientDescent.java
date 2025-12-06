package gss.optimizers;

import gss.*;
import gss.arr.*;
import java.util.*;

import static gss.Util.*;

public class GradientDescent extends Optimizer
{
	public GradientDescent()
	{
		super();
	}
	public GradientDescent(Base...prms)//, float lr)
	{
		super();
		for (Base n:prms)
			params.add(n);
	}
	public GradientDescent(ArrayList<Base>...prms)
	{
		for (ArrayList<Base> ar:prms)
			params.addAll(ar);
	}
	@Override
	public void step()
	{
		for (Base dt:params)
		{
			if (!dt.hasGradient())
				continue;
			// Base gr=dt.detachGradient();
			// int[] tmpShp=new int[dt.shape.length];
			for (int i=0;i < dt.length;i++)
			{
				float v=dt.get1d(i);
				v -= dt.get1dGrad(i) * learningRate;
				dt.set1d(i, v);
				// dt[i] -= gr[i] * learningRate;
			}
		}
		// super.update(params);
	}
}
