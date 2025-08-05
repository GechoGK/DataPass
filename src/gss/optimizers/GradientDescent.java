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
	public void update()
	{
		for (Base dt:params)
		{
			if (!dt.requiresGradient())
				continue;
			Base gr=dt.detachGradient();
			int[] tmpShp=new int[dt.shape.length];
			for (int i=0;i < dt.length;i++)
			{
				indexToShape(i, dt.shape, tmpShp);
				int ind=shapeToIndex(tmpShp, dt.shape, dt.strides);

				float v=dt.getRaw(ind);
				v += gr.getRaw(ind) * learningRate;
				dt.setRaw(ind, v);
				// dt[i] -= gr[i] * learningRate;
			}
		}
		// super.update(params);
	}
}
