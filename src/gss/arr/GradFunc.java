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
	public abstract Base backward(Base host, Base...childs, Object params)
	@Override
	public String toString()
	{
		return name + "Gradient[" + hashCode() + "]";
	}
	public static GradFunc additionGradient = new GradFunc("addition")
	{
		@Override
		public Base backward(Base host, Base[] childs, Object params)
		{
			/*
			 addition gradient
			 a + b = c
			 c.grad = 2
			 a.grad = c.grad * 1
			 b.gead = c.grad * 1
			 */
			Base a1=childs[0]; // a
			Base a2=childs[1]; // b
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
	public static GradFunc dotGradient=new GradFunc("dot"){
		@Override
		public Base backward(Base host, Base[] childs, Object params)
		{
			/*
			 // forward method.
			 for (int r=0;r < sh1[0];r++)
			 for (int c=0;c < sh2[1];c++)
			 {
			 float sum=0;
			 for (int i=0;i < sh1[1];i++)
			 sum += d1.get(r, i) * d2.get(i, c);
			 f[shapeToIndex(ar(r, c), dotShape)] = sum;
			 }
			 */
			Base a=childs[0];
			Base b=childs[1];

			

			return null;
		}
	};
}
