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
	public static GradFunc subtractionGradient = new GradFunc("subtraction")
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
					a2.setGrad(tmpShape, -host.getGrad(tmpShape));
			}
			return null;
		}
	};
	public static GradFunc dotGradient=new GradFunc("dot"){
		@Override
		public Base backward(Base host, Base[] childs, Object params)
		{
			// fix copy problem.
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
			Base a=childs[0]; // [3,2]
			Base b=childs[1]; // [2,12]
			// host [3,12];
			// System.out.println(a);
			// System.out.println(b);
			// System.out.println("= " + host);
			for (int ar=0;ar < a.shape[0];ar++) // 3
				for (int bc=0;bc < b.shape[1];bc++) // 12
				{
					// get grad at host.grad[ar][bc];
					float grd=host.getGrad(ar, bc);
					for (int ac=0;ac < a.shape[1];ac++) // 2
					{
						float av=a.get(ar, ac);
						float bv=b.get(ac, bc);
						if (a.requiresGradient())
							a.setGrad(new int[]{ar,ac}, bv * grd);
						if (b.requiresGradient())			
							b.setGrad(new int[]{ac,bc}, av * grd);			
					}
				}
			return null;
		}
	};
	public static GradFunc reshapeGradient=new GradFunc("reshape"){
		@Override
		public Base backward(Base host, Base[] childs, Object params)
		{
			// System.out.println("reshape gradient");
			return null;
		}
	};
	public static GradFunc transposeGradient=new GradFunc("transpose"){
		@Override
		public Base backward(Base host, Base[] childs, Object params)
		{
			// System.out.println("transpose gradient");
			return null;
		}
	};
	public static GradFunc copyGradient=new GradFunc("copy"){
		@Override
		public Base backward(Base host, Base[] childs, Object params)
		{
			// System.out.println("copy gradient");
			// System.out.println(a.length + " ====== " + host.length);
			Base a=childs[0];
			Base grd=host.detachGradient();
			int[] shp=new int[a.shape.length];
			for (int i=0;i < a.length;i++)
			{
				a.setGrad(indexToShape(i, a.shape, shp), grd.getRaw(i)); // avoid getRaw.
			}
			return null;
		}
	};
	public static GradFunc trimGradient=new GradFunc("trim"){
		@Override
		public Base backward(Base host, Base[] childs, Object params)
		{
			// System.out.println("trim gradient");
			return null;
		}
	};
	public static GradFunc copyToGradient=new GradFunc("copy to"){
		@Override
		public Base backward(Base host, Base[] childs, Object params)
		{
			// System.out.println("copy to gradient");
			return null;
		}
	};

}
