package gss.optimizers;

import gss.*;
import gss.arr.*;
import java.util.*;

public class GradientDescent extends Optimizer
{
	public GradientDescent()
	{
		super();
	}
	public GradientDescent(Data...prms)//, float lr)
	{
		super();
		for (Data n:prms)
			params.add(n);
	}
	public GradientDescent(ArrayList<Data>...prms)
	{
		for (ArrayList<Data> ar:prms)
			params.addAll(ar);
	}
//	@Override
//	public void update()
//	{
//		for (Data p:params)
//		{
//			if (!p.requiresGradient())
//				continue;
//			float[] dt=p.base.data.data;
//			float[] gr=p.base.data.grad;
//			for (int i=0;i < dt.length;i++)
//				dt[i] -= gr[i] * learningRate;
//		}
//		// super.update(params);
//	}
}
