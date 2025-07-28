package gss.lossfunctions;

import gss.*;
import gss.arr.*;
import gss.lossfunctions.*;
import java.util.*;

public class BCE extends LossFunc
{
	@Override
	public Data forward(Data pred, Data tar)
	{
		Data prd=pred.as1DArray();
		Data tr=tar.as1DArray();

		// Data ar=new Data(bce).setEnableGradient(pred.requiresGradient());
		// ar.setGradientFunction(bceGrad, pred, tar);
		int n = prd.length;
		float loss = 0.0f;
		final float epsilon = 1e-7f; // Avoid log(0)

		for (int i = 0; i < n; i++)
		{
			// Clip predictions to [epsilon, 1-epsilon]
			float p = Math.max(epsilon, Math.min(prd.get(i), 1 - epsilon));
			loss += tr.get(i) * Math.log(p) + (1 - tr.get(i) * Math.log(1 - p));
		}

		// Average and negate (BCE formula)
		loss = -loss / n;
		return new Data(new float[]{loss});
	}
	// backward
//	private static GradFunc bceGrad=new GradFunc("binary cross entropy"){
//		@Override
//		public Data backward(Data host, Data[] childs, Object[] params)
//		{
//			Data ch=childs[0];
//			float[] grd=host.base.data.getGrads();
//			float[] xv=ch.base.data.getData();
//			float[] trLabel=childs[1].base.data.getData();
//			float[] g=BCE.backward(grd, xv, trLabel);
//			ch.base.data.setGrad(g); // don't use this method.
//			return null;
//		}
//	};
	private static float[] backward(float[] grad, float[] x, float[] trueLabel)
	{
		int n = x.length;
		float[] gradient = new float[n];
		final float epsilon = 1e-7f; // Avoid division by 0

		for (int i = 0; i < n; i++)
		{
			// Clip predictions to [epsilon, 1-epsilon]
			float p = Math.max(epsilon, Math.min(x[i], 1 - epsilon));

			// Gradient formula: (p - y) / (p * (1 - p)) * (1/n) * grad[0]
			float grad_i = (p - trueLabel[i]) / (p * (1 - p));
			gradient[i] = (grad_i / n) * grad[0];
		}
		return gradient;
	}

	// example
//	public static void test()
//	{
//		BCE b=new BCE();
//		// Forward pass
//		float[] pred = {0.2f, 0.8f};
//		float[] trueLabel = {1.0f, 0.0f};
//		float[] loss = b.forward(pred, trueLabel); // Returns [~1.609]
//
//		print("loss =" + Arrays.toString(loss));
//		// Backward pass (grad[0] = 1.0 for loss functions)
//		float[] gradient = backward(new float[]{1.0f}, pred, trueLabel); 
//		// Returns [ (0.2-1)/(0.2*0.8)/2 ≈ -2.5, (0.8-0)/(0.8*0.2)/2 ≈ 2.5 ]
//
//		print("grad =" + Arrays.toString(gradient));
//
//		print("==== with Data ====");
//
//		Data pr=new Data(pred).setEnableGradient(true);
//		Data tr=new Data(trueLabel);
//
//		Data rs=b.forward(pr, tr);
//		print("loss", rs);
//		print("---- grad ----");
//		rs.setGrad(1);
//		rs.backward();
//		printGrad(pr);
//
//		Test1.test(Arrays.equals(rs.base.data.getData(), loss), "loss equals with Data");
//		Test1.test(Arrays.equals(pr.base.data.getGrads(), gradient), "gradient equals with Data");
//	}
}
