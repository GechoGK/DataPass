package gss.lossfunctions;

import gss.*;
import gss.arr.*;
import gss.lossfunctions.*;
import java.util.*;

public class BCE extends LossFunc
{
	@Override
	public Base forward(Base pred, Base tar)
	{
		Base prd=pred.as1DArray();
		Base tr=tar.as1DArray();

		// Data ar=new Data(bce).setEnableGradient(pred.requiresGradient());
		// ar.setGradientFunction(bceGrad, pred, tar);
		int n = prd.length;
		float loss = 0.0f;
		final float epsilon = 1e-7f; // Avoid log(0)

		for (int i = 0; i < n; i++)
		{
			// Clip predictions to [epsilon, 1-epsilon]
			float tarF=tr.get(i);
			float p = Math.max(epsilon, Math.min(prd.get(i), 1 - epsilon));
			loss += tarF * Math.log(p) + (1 - tarF * Math.log(1 - p));
		}

		// Average and negate (BCE formula)
		loss = -loss / n;
		Base b = new Base(new float[]{loss});
		b.setRequiresGradient(pred.requiresGradient());
		if (b.requiresGradient())
			b.setGradientFunction(bceGrad, pred, tar);
		return b;
	}
	// backward
	private static GradFunc bceGrad=new GradFunc("binary cross entropy"){
		@Override
		public Base backward(Base host, Base[] childs, Object params)
		{
			Base ch=childs[0]; // 1d array
			Base trueLabel=childs[1]; // 1d array
			// host = 0d array(single value)
			// float[] grd=host.base.data.getGrads();
			// float[] xv=ch.base.data.getData();
			// float[] trLabel=childs[1].base.data.getData();
			// float[] g=BCE.backward(grd, xv, trLabel);
			// ch.base.data.setGrad(g); // don't use this method.
			int n = ch.shape[0];
			// float[] gradient = new float[n];
			final float epsilon = 1e-7f; // Avoid division by 0
			for (int i = 0; i < n; i++)
			{
				// Clip predictions to [epsilon, 1-epsilon]
				float p = Math.max(epsilon, Math.min(ch.get(i), 1 - epsilon));

				// Gradient formula: (p - y) / (p * (1 - p)) * (1/n) * grad[0]
				float grad_i = (p - trueLabel.get(i)) / (p * (1 - p));
				ch.setGrad(Util.ar(i) , (grad_i / n) * host.getGrad(0));
			}
			return null;
		}
	};
//	private static float[] backward(float[] grad, float[] x, float[] trueLabel)
//	{
//		int n = x.length;
//		float[] gradient = new float[n];
//		final float epsilon = 1e-7f; // Avoid division by 0
//
//		for (int i = 0; i < n; i++)
//		{
//			// Clip predictions to [epsilon, 1-epsilon]
//			float p = Math.max(epsilon, Math.min(x[i], 1 - epsilon));
//
//			// Gradient formula: (p - y) / (p * (1 - p)) * (1/n) * grad[0]
//			float grad_i = (p - trueLabel[i]) / (p * (1 - p));
//			gradient[i] = (grad_i / n) * grad[0];
//		}
//		return gradient;
//	}

	// example
//	public static void main(String[]args)
//	{
//		BCE b=new BCE();
//		// Forward pass
//		float[] pred = {0.2f, 0.8f};
//		float[] trueLabel = {1.0f, 0.0f};
////		float[] loss = b.forward(pred, trueLabel); // Returns [~1.609]
////
////		print("loss =" + Arrays.toString(loss));
////		// Backward pass (grad[0] = 1.0 for loss functions)
////		float[] gradient = backward(new float[]{1.0f}, pred, trueLabel); 
////		// Returns [ (0.2-1)/(0.2*0.8)/2 ≈ -2.5, (0.8-0)/(0.8*0.2)/2 ≈ 2.5 ]
////
////		print("grad =" + Arrays.toString(gradient));
////
////		print("==== with Data ====");
//
//		Base pr=new Base(pred).setRequiresGradient(true);
//		Base tr=new Base(trueLabel);
//
//		Base rs=b.forward(pr, tr);
//		print("loss" + rs);
//		rs.printArray();
//		print("---- grad ----");
//		rs.fillGrad(1);
//		rs.backward();
//		pr.detachGradient().printArray();
//
//		// Test1.test(Arrays.equals(rs.base.data.getData(), loss), "loss equals with Data");
//		// Test1.test(Arrays.equals(pr.base.data.getGrads(), gradient), "gradient equals with Data");
//	}
	static void print(Object o)
	{
		System.out.println(o);
	}
}
