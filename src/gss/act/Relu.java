package gss.act;

import gss.*;
import gss.arr.*;

public class Relu extends Activation
{
	@Override
	public Base forward(Base arr)
	{
		Base dt=arr.reshape(-1);
		float[] out=new float[dt.length];
		for (int i=0;i < dt.length;i++)
		{
			out[i] = Math.max(0, dt.get(i));
		}
		Base arrOut=new Base(out, arr.shape).setRequiresGradient(arr.requiresGradient());
		// gradient in progress.
		if (arrOut.requiresGradient())
			arrOut.setGradientFunction(reluGradient, arr);
		return arrOut;
	}
	// how gradient works by this.grad or this.data ?
	public static GradFunc reluGradient=new GradFunc("relu"){
		@Override
		public Base backward(Base host, Base[] childs, Object params)
		{
			Base arr=childs[0];
			// Base gd=host.base.data.getGrads();
			// Base dt=arr.base.data.getData();
			for (int i=0;i < arr.shape[0];i++)
			{
				if (arr.get(i) > 0)
					arr.setGrad(Util.ar(i), host.getGrad(i));
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
