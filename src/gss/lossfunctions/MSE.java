package gss.lossfunctions;

import gss.*;
import gss.arr.*;
import gss.lossfunctions.*;
import java.util.*;

public class MSE extends LossFunc
{
	/*
	 problem when converting to Data.
	 */

	@Override
	public Base forward(Base pred, Base tar)
	{
//		if (pred.getLength() != tar.getLength())
//			throw new RuntimeException("unable to compute loss function with different array lengths");
		Base prd=pred.as1DArray(); // .base.data.getData();
		Base tr=tar.as1DArray(); // .base.data.getData();
//		float[] mse=forward(prd, tr);
//		Data ar=new Data(mse).setEnableGradient(pred.requiresGradient());
//		ar.setGradientFunction(mseGrad, pred, tar);
		int n = prd.length;
		float loss = 0.0f;

		// Compute sum of squared errors
		for (int i = 0; i < n; i++)
		{
			float diff = prd.get(i) - tr.get(i);
			loss += diff * diff;
		}

		// Average the sum
		loss /= n;

		// Return loss as a 1-element array
		Base b = new Base(new float[]{loss});
		b.setRequiresGradient(pred.requiresGradient());
		if (b.requiresGradient())
			b.setGradientFunction(mseGrad, pred, tar);
		return b;
	}
	private static GradFunc mseGrad=new GradFunc("mean squared error"){
		@Override
		public Base backward(Base host, Base[] childs, Object params)
		{
			Base prd=childs[0]; // 1d array
			Base trueLabel = childs[1]; // 1d array
			// host = pd array(single value)
			// float[] grd=host.base.data.getGrads();
			// float[] xv=ch.base.data.getData();
			// float[] trLabel=childs[1].base.data.getData();
			// float[] g=MSE.backward(grd, xv, trLabel);
			// ch.base.data.setGrad(g); // don't use this method.
			int n = prd.length;
			// float[] gradient = new float[n];
			// Compute gradient for each prediction
			for (int i = 0; i < n; i++)
			{
				// Gradient formula: 2 * (x[i] - trueLabel[i]) / n
				prd.setGrad(Util.ar(i), (2 * (prd.get(i) - trueLabel.get(i)) / n) * host.getGrad(0));
			}

			return null;
		}
	};
	// Class-level variable to store true labels from forward pass
//	private static float[] backward(float[] grad, float[] x, float[] trueLabel)
//	{
//		int n = x.length;
//		float[] gradient = new float[n];
//
//		// Compute gradient for each prediction
//		for (int i = 0; i < n; i++)
//		{
//			// Gradient formula: 2 * (x[i] - trueLabel[i]) / n
//			gradient[i] = 2 * (x[i] - trueLabel[i]) / n;
//
//			// Multiply by upstream gradient (grad[0] = 1 for loss functions)
//			gradient[i] *= grad[0];
//		}
//
//		return gradient;
//	}
	/// example
//	public static void main(String...g)
//	{
//		// Forward pass
//		MSE m=new MSE();
//		float[] pred = {1.0f, 5.0f};
//		float[] trueLabel = {3.0f, 5.0f};
//		// float[] loss = m.forward(pred, trueLabel); // Returns [6.5]
//		// print("loss =" + Arrays.toString(loss));
//
//		// Backward pass
////		float[] gradFromAbove = {1.0f}; // Upstream gradient (dL/dL = 1)
////		float[] gradient = backward(gradFromAbove, pred, trueLabel); // Returns [2.0, 3.0]
////
////		print("grad =" + Arrays.toString(gradient));
////
////		print("==== with Data ====");
//
//		Base pr=new Base(pred).setRequiresGradient(true);
//		Base tr=new Base(trueLabel);
//
//		Base rs=m.forward(pr, tr);
//		System.out.println("loss :" + rs.get(0));
//		System.out.println("---- grad ----");
//		rs.fillGrad(1);
//		rs.backward();
//		pr.detachGradient().printArray();
//
//		// Test1.test(Arrays.equals(rs.base.data.getData(), loss), "loss equals with Data");
//		// Test1.test(Arrays.equals(pr.base.data.getGrads(), gradient), "gradient equals with Data");
//	}
}
