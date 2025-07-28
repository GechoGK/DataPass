package gss.lossfunctions;

import gss.*;
import gss.arr.*;
import gss.lossfunctions.*;
import java.util.*;

public class MAE extends LossFunc
{
	@Override
	public Data forward(Data pred, Data tar)
	{	
		Data prd=pred.as1DArray(); // .base.data.getData();
		Data tr=tar.as1DArray(); // .base.data.getData();
		// float[] mae=forward(prd, tr);
		// Data ar=new Data(mae).setEnableGradient(pred.requiresGradient());
		// ar.setGradientFunction(maeGrad, pred, tar);
		// return ar;
		int n = prd.length;
		float loss = 0.0f;

		// Sum of absolute differences
		for (int i = 0; i < n; i++)
		{
			loss += Math.abs(prd.get(i) - tr.get(i));
		}

		// Average the sum
		loss /= n;

		// Return loss as a 1-element array
		return new Data(new float[]{loss});
	}
//	private static GradFunc maeGrad=new GradFunc("mean absolute error"){
//
//		@Override
//		public Data backward(Data host, Data[] childs, Object[] params)
//		{
//			Data ch=childs[0];
//			float[] grd=host.base.data.getGrads();
//			float[] xv=ch.base.data.getData();
//			float[] trLabel=childs[1].base.data.getData();
//			float[] g=MAE.backward(grd, xv, trLabel);
//			ch.base.data.setGrad(g); // don't use this method.
//			return null;
//		}
//	};
	// backward
	private static float[] backward(float[]grad, float[] x, float[] trueLabel)
	{
		int n = x.length;
		float[] gradient = new float[n];

		// Compute gradient for each prediction
		for (int i = 0; i < n; i++)
		{
			// Gradient formula: sign(x[i] - trueLabel[i]) / n
			float diff = x[i] - trueLabel[i];
			gradient[i] = (float) (Math.signum(diff) / n) * grad[0];
		}

		return gradient;
	}

	// example

//	public static void test()
//	{
//		// Forward pass
//		MAE m=new MAE();
//		float[] pred = {3.0f, 5.0f};
//		float[] trueLabel = {1.0f, 2.0f};
//		float[] loss = m.forward(pred, trueLabel); // Returns [2.5]
//		print("loss =" + Arrays.toString(loss));
//		// Backward pass
//		float[] gradFromAbove = {1.0f};
//		float[] gradient = backward(gradFromAbove, pred, trueLabel); // Returns [1.0/2=0.5, 1.0/2=0.5]
//
//		print("grad =" + Arrays.toString(gradient));
//
//		print("==== with Data ====");
//
//		Data pr=new Data(pred).setEnableGradient(true);
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
}
