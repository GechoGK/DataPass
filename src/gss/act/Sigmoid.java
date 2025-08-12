package gss.act;

import gss.*;
import gss.arr.*;

public class Sigmoid extends Activation
{
	/*
	 float sigmoidForward(float x) {
	 return 1.0f / (1.0f + (float) Math.exp(-x));
	 }
	 */
	@Override
	public Base forward(Base array)
	{
		Base dt=array.reshape(-1);
		float[] out=new float[dt.length];
		for (int i=0;i < dt.length;i++)
		{
			out[i] = 1f / (1f + (float)Math.exp(-dt.get(dt.indexToShape(i))));
		}
		Base arrOut=new Base(out, array.shape).setRequiresGradient(dt.requiresGradient());
		// gradient in progress.
		if (arrOut.requiresGradient())
			arrOut.setGradientFunction(sigmoidGradient, dt);
		return arrOut;
	}
	public static GradFunc sigmoidGradient=new GradFunc("sigmoid"){

		/*
		 float sigmoidBackward(float grad, float x) {
		 float sigmoid_x = 1.0f / (1.0f + (float) Math.exp(-x)); // Recompute σ(x)
		 return grad * sigmoid_x * (1.0f - sigmoid_x); // Chain rule: grad * σ'(x)
		 }
		 */
		@Override
		public Base backward(Base host, Base[] childs, Object params)
		{
			// cache sig. for performance
			Base a1=childs[0];
			// float[] grd=host.base.data.getGrads();
			// float[] dt=a1.base.data.getData();
			for (int i=0;i < host.length;i++)
			{
				int sh[]=a1.indexToShape(i);
				float sig = 1f / (1f + (float)Math.exp(-a1.get(sh)));
				a1.setGrad(sh, host.getGrad(host.indexToShape(i)) * sig * (1 - sig));
			}
			// childs[0].base.data.setGrad(dt);
			return null;
		}
	};
//	public static Value sigmoid(Value op1)
//	{
//		/*
//		 def sigmoid(self, x):
//		 return 1 / (1 + np.exp(-x))
//
//		 def sigmoid_derivative(self, x):
//		 return x * (1 - x)
//		 */
//		double out = 1 / (1 + Math.exp(-op1.data));
//		Value v=new Value(out, op1){
//			@Override
//			public void backward()
//			{
//				this.childs[0].setGrad(this.grad * (1 - this.grad));
//				super.backward();
//			}
//		};
//		return v;
//	}
}
