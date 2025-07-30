package gss.lossfunctions;

import gss.*;
import gss.arr.*;
import gss.lossfunctions.*;
import java.util.*;

public class MCCE extends LossFunc
{
	// it uses softmax inside it's function.
	@Override
	public Base forward(Base pred, Base tar)
	{
		Base prd=pred.as1DArray(); // .base.data.getData();
		Base tr=tar.as1DArray(); // .base.data.getData();
//		float[] mcce=forward(prd, tr);
//		Data ar=new Data(mcce).setEnableGradient(pred.requiresGradient());
//		ar.setGradientFunction(mcceGrad, pred, tar);
//
		int n = prd.length; // Number of classes
		float loss = 0.0f;
		final float epsilon = 1e-7f; // Avoid log(0)

		// Compute softmax probabilities (from logits)
		float[] softmax = new float[n];
		float expSum = 0.0f;

		// Exponentiate and sum
		for (int i = 0; i < n; i++)
		{
			softmax[i] = (float) Math.exp(prd.get(i));
			expSum += softmax[i];
		}

		// Normalize to probabilities
		for (int i = 0; i < n; i++)
		{
			softmax[i] /= expSum;
		}

		// Compute cross-entropy loss
		for (int i = 0; i < n; i++)
		{
			// Clip to avoid log(0) and use one-hot labels
			loss += tr.get(i) * Math.log(Math.max(epsilon, softmax[i]));
		}
		loss = -loss; // Negate and return as scalar

		return new Base(new float[]{loss});
	}
	// backward
//	private static GradFunc mcceGrad=new GradFunc(""){
//		@Override
//		public Data backward(Data host, Data[] childs, Object[] params)
//		{
//			Data ch=childs[0];
//			float[] grd=host.base.data.getGrads();
//			float[] xv=ch.base.data.getData();
//			float[] trLabel=childs[1].base.data.getData();
//			float[] g=MCCE.backward(grd, xv, trLabel);
//			ch.base.data.setGrad(g); // don't use this method.
//			return null;
//		}
//	};

	private static float[] backward(float[] grad, float[] x, float[] trueLabel)
	{
		int n = x.length; // Number of classes
		float[] gradient = new float[n];

		// Recompute softmax during backward pass
		float[] softmax = new float[n];
		float expSum = 0.0f;
		for (int i = 0; i < n; i++)
		{
			softmax[i] = (float) Math.exp(x[i]);
			expSum += softmax[i];
		}
		for (int i = 0; i < n; i++)
		{
			softmax[i] /= expSum;
		}

		// Gradient formula: (softmax - trueLabel) * grad[0]
		for (int i = 0; i < n; i++)
		{
			gradient[i] = (softmax[i] - trueLabel[i]) * grad[0];
		}

		return gradient;
	}


	// example
//	public static void test()
//	{
//		MCCE m=new MCCE();
//		// Forward pass
//		float[] logits = {2.0f, 1.0f, 0.1f}; // Logits (pre-softmax)
//		float[] trueLabel = {1.0f, 0.0f, 0.0f}; // One-hot encoded (class 0)
//		float[] loss = m.forward(logits, trueLabel); 
//		// Softmax ≈ [0.659, 0.242, 0.099], loss ≈ -log(0.659) ≈ 0.417
//		print("loss =" + Arrays.toString(loss));
//
//		// Backward pass
//		float[] gradient = backward(new float[]{1.0f}, logits, trueLabel);
//		// gradient ≈ [0.659 - 1 = -0.341, 0.242 - 0 = 0.242, 0.099 - 0 = 0.099]
//
//		print("grad =" + Arrays.toString(gradient));
//
//		print("==== with Data ====");
//
//		Data pr=new Data(logits).setEnableGradient(true);
//		Data tr=new Data(trueLabel);
//
//		Data rs=m.forward(pr, tr);
//		print("loss", rs);
//		print("---- grad ----");
//		rs.setGrad(1);
//		rs.backward();
//		printGrad(pr);
//
//		Test1.test(Arrays.equals(rs.base.data.getData(), loss), "loss equals with Data");
//		Test1.test(Arrays.equals(pr.base.data.getGrads(), gradient), "gradient equals with Data");
//	}

	/*
	 // softmax as a separate layer.
	 float[] forward(float[] probabilities, float[] trueLabel) {
	 // Directly compute cross-entropy on probabilities
	 float loss = 0.0f;
	 for (int i = 0; i < probabilities.length; i++) {
	 loss += trueLabel[i] * Math.log(probabilities[i]);
	 }
	 return new float[]{-loss};
	 }

	 float[] backward(float[] grad, float[] probabilities, float[] trueLabel) {
	 // Gradient: (probabilities - trueLabel) * grad[0]
	 float[] gradient = new float[probabilities.length];
	 for (int i = 0; i < gradient.length; i++) {
	 gradient[i] = (probabilities[i] - trueLabel[i]) * grad[0];
	 }
	 return gradient;
	 }
	 */
}
