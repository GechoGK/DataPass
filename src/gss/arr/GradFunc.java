package gss.arr;

import static gss.Util.*;

public abstract class GradFunc
{
	// name for debugging purpose.
	private String name;

	public GradFunc()
	{
		this.name = "unknown";
	}
	public GradFunc(String name)
	{
		this.name = name;
	}
	public abstract Data backward(Data host, Data...childs, Object params)
	@Override
	public String toString()
	{
		return name + "Gradient[" + hashCode() + "]";
	}
	public static GradFunc additionGradient = new GradFunc("addition")
	{
		@Override
		public Data backward(Data host, Data[] childs, Object params)
		{
			/*
			 addition gradient
			 a + b = c
			 c.grad = 2
			 a.grad = c.grad * 1
			 b.gead = c.grad * 1
			 */
			Data a1=childs[0]; // a
			Data a2=childs[1]; // b
			int[] shape=host.shape;
			int[] tmpShape=new int[shape.length];
			for (int i=0;i < host.length;i++)
			{
				indexToShape(i, shape, tmpShape);
				if (a1.requiresGradient())
					a1.setGrad(tmpShape, host.getGrad(tmpShape));
				if (a2.requiresGradient())
					a2.setGrad(tmpShape, host.getGrad(tmpShape));
			}
			return null;
		}
	};
}
