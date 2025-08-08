package gss.lossfunctions;

import gss.*;
import gss.arr.*;
import gss.lossfunctions.*;
import java.util.*;

public class MCCE extends LossFunc
{
	// MCCE = (multi label cross entropy).
	// no softmax is used.
	@Override
	public Base forward(Base pred, Base tar)
	{
		Base prd=pred.as1DArray(); // .base.data.getData();
		Base tr=tar.as1DArray(); // .base.data.getData();

		int n = prd.length; // Number of classes
		float loss = 0.0f;
		final float epsilon = 1e-7f; // Avoid log(0)

		// Compute cross-entropy loss
		for (int i = 0; i < n; i++)
		{
			// Clip to avoid log(0) and use one-hot labels
			loss += tr.get(i) * Math.log(Math.max(epsilon, prd.get(i)));
		}
		loss = -loss; // Negate and return as scalar

		Base b = new Base(new float[]{loss});
		b.setRequiresGradient(pred.requiresGradient());
		if (b.requiresGradient())
			b.setGradientFunction(mcceGrad, prd, tr);
		return b;
	}
	// backward
	private static GradFunc mcceGrad=new GradFunc(""){
		@Override
		public Base backward(Base host, Base[] childs, Object params)
		{
			Base prd=childs[0]; // 1d array
			Base trueLabel=childs[1]; // 1d array
			// host = 0d array(single value)
			int n = prd.length; // Number of classes

			// Gradient formula: (softmax - trueLabel) * grad[0]
			for (int i = 0; i < n; i++)
			{
				prd.setGrad(Util.ar(i), (prd.get(i) - trueLabel.get(i)) * host.getGrad(0));
			}

			// return gradient;
			return null;
		}
	};
	// example
//	public static void main(String[]args)
//	{
//		MCCE m=new MCCE();
//		// Forward pass
//		float[] logits = {1.0f, 5.0f, 1.0f}; // Logits (pre-softmax)
//		float[] trueLabel = {2.0f, 1.0f, 2.0f}; // One-hot encoded (class 0)
////		float[] loss = m.forward(logits, trueLabel); 
////		// Softmax ≈ [0.659, 0.242, 0.099], loss ≈ -log(0.659) ≈ 0.417
////		System.out.println("loss =" + Arrays.toString(loss));
////
////		// Backward pass
////		float[] gradient = m.backward(new float[]{1.0f}, logits, trueLabel);
////		// gradient ≈ [0.659 - 1 = -0.341, 0.242 - 0 = 0.242, 0.099 - 0 = 0.099]
////
////		System.out.println("grad =" + Arrays.toString(gradient));
////
////		print("==== with Data ====");
//
//		Base pr=new Base(logits).setRequiresGradient(true);
//		Base tr=new Base(trueLabel);
//
//		Base rs=m.forward(pr, tr);
//		System.out.println("loss " + rs);
//		rs.printArray();
//		System.out.println("---- grad ----");
//		rs.fillGrad(1);
//		rs.backward();
//		pr.detachGradient().printArray();
//		
//		// Test1.test(Arrays.equals(rs.base.data.getData(), loss), "loss equals with Data");
//		// Test1.test(Arrays.equals(pr.base.data.getGrads(), gradient), "gradient equals with Data");
//	}
//
	//*
	// softmax as a separate layer.
//	float[] forward(float[] probabilities, float[] trueLabel)
//	{
//		// Directly compute cross-entropy on probabilities
//		float loss = 0.0f;
//		for (int i = 0; i < probabilities.length; i++)
//		{
//			loss += trueLabel[i] * Math.log(probabilities[i]);
//		}
//		return new float[]{-loss};
//	}
//	float[] backward(float[] grad, float[] probabilities, float[] trueLabel)
//	{
//		// Gradient: (probabilities - trueLabel) * grad[0]
//		float[] gradient = new float[probabilities.length];
//		for (int i = 0; i < gradient.length; i++)
//		{
//			gradient[i] = (probabilities[i] - trueLabel[i]) * grad[0];
//		}
//		return gradient;
//	}
	//  */
}
