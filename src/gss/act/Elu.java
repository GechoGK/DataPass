package gss.act;

import gss.*;
import gss.arr.*;

public class Elu extends Activation
{
	private float eps;
	public Elu(float eps)
	{
		this.eps = eps;
	}
	public Elu()
	{
		this.eps = 0.1f;
	}
	@Override
	public Base forward(Base array)
	{
		Base dt=array.reshape(-1);
		float[] out=new float[dt.length];
		for (int i=0;i < dt.length;i++)
		{
			float v=dt.get(dt.indexToShape(i));
			out[i] = v >= 0 ? dt.get(dt.indexToShape(i)): v * eps;
		}
		Base arrOut=new Base(out, array.shape).setRequiresGradient(dt.hasGradient());
		arrOut.setGradientFunctionS(reluGradient, eps, dt);
		return arrOut;
	}
	// how gradient works by this.grad or this.data ?
	public static GradFunc reluGradient=new GradFunc("relu"){
		@Override
		public Base backward(Base host, Base[] childs, Object params)
		{
			Base arr=childs[0];
			float eps=params;
			// Base gd=host.base.data.getGrads();
			// Base dt=arr.base.data.getData();
			for (int i=0;i < arr.length;i++)
			{
				float v=arr.get(arr.indexToShape(i));
				float g=host.getGrad(host.indexToShape(i));
				if (v >= 0)
					arr.setGrad(arr.indexToShape(i), g);
				else
					arr.setGrad(arr.indexToShape(i), g * eps);
			}
			return null;
		}
	};
//	public static Value relu(Value op1)
//	{
//		/*
//		 def relu(self, x):
//		 return Math.max(0,x);
//		 def relu_derivative(self, x):
//		 return x<0?0:1;
//		 */
//		double out = Math.max(0, op1.data);
//		Value v=new Value(out, op1){
//			@Override
//			public void backward()
//			{
//				this.childs[0].setGrad(this.grad < 0 ? 0 : 1);
//				super.backward();
//			}
//		};
//		return v;
//	}
}
