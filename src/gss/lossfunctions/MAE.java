package gss.lossfunctions;

import gss.*;
import gss.arr.*;
import gss.optimizers.*;

import static gss.Util.*;

public class MAE extends LossFunc
{
	@Override
	public Base forward(Base predicted, Base target)
	{	
		Base prd=predicted.as1DArray(); // .base.data.getData();
		Base tr=target.as1DArray(); // .base.data.getData();
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
		Base b = new Base(new float[]{loss});
		b.setRequiresGradient(prd.hasGradient());
		if (b.hasGradient())
			b.setGradientFunction(maeGrad, prd, tr);
		return b;
	}
	private static GradFunc maeGrad=new GradFunc("mean absolute error"){
		@Override
		public Base backward(Base host, Base[] childs, Object params)
		{
			Base ch=childs[0]; // 1d array
			Base trueLabel=childs[1]; // 1d array
			// host = 0d array(single value)
			// float[] grd=host.base.data.getGrads();
			// float[] xv=ch.base.data.getData();
			// float[] trLabel=childs[1].base.data.getData();
			// float[] g=MAE.backward(grd, xv, trLabel);
			int n = ch.length;
			// float[] gradient = new float[n];
			// Compute gradient for each prediction
			for (int i = 0; i < n; i++)
			{
				// Gradient formula: sign(x[i] - trueLabel[i]) / n
				float diff = ch.get(i) - trueLabel.get(i);
				ch.setGrad(Util.ar(i),  (float) (Math.signum(diff) / n) * host.getGrad(0));
			}

			// ch.base.data.setGrad(g); // don't use this method.
			return null;
		}
	};
	// backward
//	private static float[] backward(float[]grad, float[] x, float[] trueLabel)
//	{
//		int n = x.length;
//		float[] gradient = new float[n];
//
//		// Compute gradient for each prediction
//		for (int i = 0; i < n; i++)
//		{
//			// Gradient formula: sign(x[i] - trueLabel[i]) / n
//			float diff = x[i] - trueLabel[i];
//			gradient[i] = (float) (Math.signum(diff) / n) * grad[0];
//		}
//
//		return gradient;
//	}
	// example

//	public static void main(String[]args) throws Exception
//	{
//		// Forward pass
//		MAE m=new MAE();
//		float[][] pred = {{.11f, 0.98f},{.99f,.34f}};
//		float[][] trueLabel = {{.9f, 0.30f},{.1f,.999f}};
//
////		float[] loss = m.forward(pred, trueLabel); // Returns [2.5]
////		print("loss =" + Arrays.toString(loss));
////		// Backward pass
////		float[] gradFromAbove = {1.0f};
////		float[] gradient = backward(gradFromAbove, pred, trueLabel); // Returns [1.0/2=0.5, 1.0/2=0.5]
////
////		print("grad =" + Arrays.toString(gradient));
////
////		print("==== with Data ====");
////
//		Base pr=NDArray.wrap(flatten(pred), 2, 2).setRequiresGradient(true);
//		Base tr=new Base(flatten(trueLabel), 2, 2);
//
//		GradientDescent gd=new GradientDescent(pr);
//
//		float loss=10;
//		while (loss >= 0.001f)
//		{
//			Base rs=m.forward(pr, tr);
//			loss = rs.get(0);
//			print(loss);
//			// rs.printArray();
//			// System.out.println("---- grad ----");
//			rs.fillGrad(1);
//			rs.backward();
//
//			// pr.detachGradient().printArray();
//			gd.step();
//			gd.zeroGrad();
//			// Thread.sleep(100);
//		}
//		print(line(20));
//		print("loss :" + loss);
//		pr.printArray();
//		// Test1.test(Arrays.equals(rs.base.data.getData(), loss), "loss equals with Data");
//		// Test1.test(Arrays.equals(pr.base.data.getGrads(), gradient), "gradient equals with Data");
//	}
}
