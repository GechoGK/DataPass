package gss;

import gss.arr.*;
import java.util.*;

public class Optimizer
{
	public float learningRate=0.01f;
	public ArrayList<Base> params=new ArrayList<>();

	public Optimizer()
	{
		this(0.01f);
	}
	public Optimizer(float lr)
	{
		this.learningRate = lr;
	}
	public void update()
	{
		// update the parameters based on their gradient.
	}
	public void zeroGrad()
	{
		for (Base p:params)
		{
			if (p.requiresGradient())
				p.zeroGrad();
			// zero the gradient data.
		}
	}
}
