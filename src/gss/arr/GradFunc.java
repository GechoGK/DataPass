package gss.arr;

import gss.*;
import java.io.*;
import java.util.*;

import static gss.Util.*;

public abstract class GradFunc implements Serializable
{
	public static final long serialVersionUID=7265282763l;
	// name for debugging purpose.
	public String name;

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
				float f=host.getGrad(tmpShape);
				if (a1.hasGradient())
					a1.setGrad(tmpShape, f);
				if (a2.hasGradient())
					a2.setGrad(tmpShape, f);
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
				if (a1.hasGradient())
					a1.setGrad(tmpShape, host.getGrad(tmpShape));
				if (a2.hasGradient())
					a2.setGrad(tmpShape, -host.getGrad(tmpShape));
			}
			return null;
		}
	};
	public static GradFunc multiplicationGradient = new GradFunc("multiplication")
	{
		@Override
		public Base backward(Base host, Base[] childs, Object params)
		{
			Base a1=childs[0]; // first operand..
			Base a2=childs[1]; // second operand.
			/* for operand 1.
			 a * b = c
			 2 * 3 = 6;
			 gradient calculaion for multiplication.
			 c.grad = 5;
			 a.grad = c.grad * b.data
			 a.grad = 5 * 3 = 15;
			 b.grad = c.grad * a.data;
			 b.grad = 5 * 2 = 10
			 */
			int[] shape=host.shape;
			int[] tmpShape=new int[shape.length];
			for (int i=0;i < host.length;i++)
			{
				indexToShape(i, shape, tmpShape);
				float grad=host.getGrad(tmpShape);
				if (a1.hasGradient())
					a1.setGrad(tmpShape, grad * a2.get(tmpShape));
				if (a2.hasGradient())
					a2.setGrad(tmpShape, grad * a1.get(tmpShape));
			}
			return null;
		}
	};
	public static GradFunc divisionGradient = new GradFunc("division"){
		@Override
		public Base backward(Base host, Base[] childs, Object params)
		{
			Base a1=childs[0]; // == a  first operand.
			Base a2=childs[1]; // == b  second operand.
			// host  ==  c
			/*
			 gradient calculation for division.
			 c.grad = 5;
			 a.grad = c.grad * 1 / b.data
			 a.grad = 5 * 1 / 3 = 1.6666...
			 b.grad = -c.grad * a.value / (b.value^2);
			 b.grad = -5 * 6 / (3 * 3 ) = -3.3333...
			 */
			int[] shape=host.shape;
			int[] tmpShape=new int[shape.length];
			for (int i=0;i < host.length;i++)
			{
				indexToShape(i, shape, tmpShape);
				float grad = host.getGrad(tmpShape);
				float bVal = a2.get(tmpShape);
				if (a1.hasGradient())
					a1.setGrad(tmpShape, grad * 1 / bVal);
				if (a2.hasGradient())
					a2.setGrad(tmpShape, -grad * a1.get(tmpShape) / (bVal * bVal));
			}
			return null;
		}
	};
	public static GradFunc powGradient = new GradFunc("power"){
		@Override
		public Base backward(Base host, Base[] childs, Object params)
		{
			/*
			 pow gradient
			 example.
			 c = a ** b  // a the power of b.
			 // grad ==
			 c.grad = 2;
			 a.grad = c.grad * b.data * ( a.data ** (b.data - 1))
			 b.grad = c.grad * ( a.data ** b.data ) * log(a.data)
			 */
			Base a1=childs[0];
			Base a2=childs[1];
			int[] shape=host.shape;
			int[] tmpShape=new int[shape.length];
			for (int i=0;i < host.length;i++)
			{
				indexToShape(i, shape, tmpShape);
				float a=a1.get(tmpShape); // a.data
				float b=a2.get(tmpShape); // b.data
				if (a1.hasGradient())
					a1.setGrad(tmpShape, b * host.getGrad(tmpShape) * (float)Math.pow(a, b - 1));
				if (a2.hasGradient())
					a2.setGrad(tmpShape, host.getGrad(tmpShape) * (float)Math.pow(a, b) * (float)Math.log(a));
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
			// print("...", host.shape);
			// host [3,12];
			// print(a);
			// print(b);
			// System.out.println("= " + host);
			for (int ar=0;ar < a.shape[0];ar++) // 3
				for (int br=0;br < b.shape[0];br++) // 12
				{
					// get grad at host.grad[ar][bc];
					float grd=host.getGrad(ar, br);
					for (int ac=0;ac < a.shape[1];ac++) // 2
					{
						float av=a.get(ar, ac);
						float bv=b.get(br, ac);
						if (a.hasGradient())
							a.setGrad(new int[]{ar,ac}, bv * grd);
						if (b.hasGradient())			
							b.setGrad(new int[]{br,ac}, av * grd);			
					}
				}
			return null;
		}
	};
	public static GradFunc convolveGradient=new GradFunc("convolve"){
		@Override
		public Base backward(Base host, Base[] childs, Object params)
		{
			Base in=childs[0]; // 1d array.
			Base kr=childs[1]; // 1d array.

			// problem when the kernel is larger than the input data.
			// kernel gradient.(validCorrelation();
			NDArray.correlate1dValid(host.detachGradient(), in, kr.detachGradient());
			// input gradient.(fullCorrelation(o.grad,kr.item)
			NDArray.correlate1dFull(host.detachGradient(), kr, in.detachGradient());
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
			int[] shp=new int[a.shape.length];
			for (int i=0;i < a.length;i++)
			{
				a.setGrad(indexToShape(i, a.shape, shp), host.get1dGrad(i)); // avoid getRaw.
			}
			return null;
		}
	};
	public static GradFunc stepGradient=new GradFunc("step"){
		@Override
		public Base backward(Base host, Base[] childs, Object params)
		{
			// System.out.println("trim gradient");
			return null;
		}
	};
	public static GradFunc sumGradient=new GradFunc("sum"){
		@Override
		public Base backward(Base host, Base[] childs, Object params)
		{
			// do something.
			if (params == null)
			{
				// no dim found pass all host gradients to child.
				Base a=childs[0];
				if (!a.hasGradient())
					return null;
				float g=host.get1dGrad(0); // gradient
				for (int i=0;i < a.length;i++)
				{
					a.set1dGrad(i, g);
				}
			}
			else
			{
				Base a=childs[0];
				if (!a.hasGradient())
					return null;
				int[] axis=(int[])params;
				// print("axis for sumGradient :", axis);
				int[] newShape=copy(a.shape);
				newShape = fromNonAxis(newShape, true, axis);
				Base g=host.reshape(newShape);
				a.detachGradient().append(g.detachGradient());
			}
			return null;
		}
	};
	public static GradFunc logGradient=new GradFunc("log"){
		@Override
		public Base backward(Base host, Base[] childs, Object params)
		{
			Base a1=childs[0]; // a
			if (!a1.hasGradient())
				return null;
			int[] shape=host.shape;
			int[] tmpShape=new int[shape.length];
			for (int i=0;i < host.length;i++)
			{
				indexToShape(i, shape, tmpShape);
				a1.setGrad(tmpShape, host.getGrad(tmpShape) / a1.get(tmpShape));
			}
			return null;
		}
	};
	public static GradFunc expGradient=new GradFunc("exp"){
		@Override
		public Base backward(Base host, Base[] childs, Object params)
		{
			Base a1=childs[0]; // a
			if (!a1.hasGradient())
				return null;
			int[] shape=host.shape;
			int[] tmpShape=new int[shape.length];
			for (int i=0;i < host.length;i++)
			{
				indexToShape(i, shape, tmpShape);
				a1.setGrad(tmpShape, host.getGrad(tmpShape) * host.get(tmpShape));
			}
			return null;
		}
	};
	public static GradFunc sqrtGradient=new GradFunc("sqrt"){
		@Override
		public Base backward(Base host, Base[] childs, Object params)
		{
			// d/(2*sqrt(x))
			Base a1=childs[0]; // a
			if (!a1.hasGradient())
				return null;
			int[] shape=host.shape;
			int[] tmpShape=new int[shape.length];
			for (int i=0;i < host.length;i++)
			{
				indexToShape(i, shape, tmpShape);
				a1.setGrad(tmpShape, (float)(host.getGrad(tmpShape) / (2 * Math.sqrt(a1.get(tmpShape)))));
			}
			return null;
		}
	};
	public static GradFunc pass1Gradient=new GradFunc("pass"){
		@Override
		public Base backward(Base host, Base[] childs, Object params)
		{
			if (childs.length != 1)
				error("expected 1 child, found " + childs.length);
			Base a1=childs[0]; // a
			if (!a1.hasGradient())
				return null;
			int[] shape=host.shape;
			int[] tmpShape=new int[shape.length];
			for (int i=0;i < host.length;i++)
			{
				indexToShape(i, shape, tmpShape);
				float f=host.getGrad(tmpShape);
				a1.setGrad(tmpShape, f);
			}
			return null;
		}	
	};
	public static GradFunc pass2Gradient=new GradFunc("pass"){
		@Override
		public Base backward(Base host, Base[] childs, Object params)
		{
			if (childs.length != 1)
				error("expected 2 childs, found " + childs.length);
			Base a1=childs[0]; // a
			Base a2=childs[1]; // b
			int[] shape=host.shape;
			int[] tmpShape=new int[shape.length];
			for (int i=0;i < host.length;i++)
			{
				indexToShape(i, shape, tmpShape);
				float f=host.getGrad(tmpShape);
				if (a1.hasGradient())
					a1.setGrad(tmpShape, f);
				if (a2.hasGradient())
					a2.setGrad(tmpShape, f);
			}
			return null;
		}	
	};
	public static GradFunc indexGradient=new GradFunc("index"){
		@Override
		public Base backward(Base host, Base[] childs, Object params)
		{
			Base ch=childs[0];
			int[]inds=(int[])params;
			if (host.length != inds.length)
				error("invalid gradient value for indexes");
			for (int i=0;i < inds.length;i++)
			{
				float g=host.get1dGrad(i);
				ch.set1dGrad(inds[i], g);
			}
			return null;
		}
	};
	public static GradFunc maskGradient=new GradFunc("mask"){
		@Override
		public Base backward(Base host, Base[] childs, Object params)
		{
			// print("mask gradient not implemented.");
			Base grd=host.detachGradient().as1DArray();// .base.data.getGrads();
			int[] mask=((int[])params);
			Base c1=childs[0];
			if (host.length != c1.length)
				throw new RuntimeException("unable to compute dropout backpropagation.: invalid array length between the host and the child.");
			Base rs=NDArray.mul(grd, new Base(Util.asFloat(mask)));
			c1.setGrad(rs);
			return null;
		}
	};
	public static GradFunc concatGradient=new GradFunc("concat"){
		@Override
		public Base backward(final Base host, Base[] childs, Object params)
		{
			// error("concatination doesn't have backward method. please review the code..");
			final Base b1=childs[0];
			final Base b2=childs[1];
			int ax=params;
			{
				int diff1=Math.abs(b1.shape.length - b2.shape.length);
				final int axis=ax + diff1;
				int[]outShape1=getCommonShapeExcept(b1.shape, b2.shape, axis);

				final int[] i1_shape=copy(outShape1);
				final int[] i2_shape=copy(outShape1);
				copyB(b1.shape, i1_shape);
				copyB(b2.shape, i2_shape);

				final int diff2=i1_shape[axis];

				Util.loop(host, new Functions.ArrayConsumer(){
						@Override
						public void consume(int[] p1)
						{
							float v=host.getGrad(p1);
							if (p1[axis] >= i1_shape[axis])
							{
								p1[axis] = p1[axis] - diff2;
								// set to b2
								b2.setGrad(p1, v);
							}
							else
							// set to b1
								b1.setGrad(p1, v);
						}
					});
			}
			return null;
		}
	};
	public static GradFunc itemGradient = new GradFunc("item"){
		@Override
		public Base backward(Base host, Base[] childs, Object params)
		{
			// iterate over each stores Value classes and then call backward on them.
			if (!host.hasGradient())
				return null;
			// System.out.println("== .." + host);
			HashSet<Value> tmpLst=new HashSet<>();
			HashSet<Value> lst=new HashSet<>();
			for (int i=0;i < host.length;i++)
				lst.add(host.getValue(host.indexToShape(i)));
			while (lst.size() != 0)
			{
				// System.out.println(lst.size() + " value items found");
				for (Value v:lst)
				{
					v.backward();
					tmpLst.addAll(v.args);
				}
				lst.clear();
				lst.addAll(tmpLst);
				tmpLst.clear();
			}
			return null;
		}
	};
}
