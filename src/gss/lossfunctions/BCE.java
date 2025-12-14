package gss.lossfunctions;

import gss.*;
import gss.arr.*;
import gss.lossfunctions.*;
import java.util.*;
import gss.optimizers.*;

import static gss.Util.*;

public class BCE extends LossFunc
{
	/*
	 BCE us used for single output which will be 0 or 1.
	 example.
	 input  = [0];
	 target = [1];
	 use optimizer to update weights inorder to get result like target.
	 the input to the BCE should be normalized(sigmoid,tanh)(-1,1), if not it may have unexpected results.
	 */
	@Override
	public Base forward(Base predicted, Base target)
	{
		Base prd=predicted.as1DArray();
		Base tr=target.as1DArray();

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
			loss += tarF * Math.log(p) + (1 - tarF) * Math.log(1 - p);
		}

		// Average and negate (BCE formula)
		loss = -(loss / n);
		Base b = new Base(new float[]{loss});
		b.setRequiresGradient(prd.hasGradient());
		if (b.hasGradient())
			b.setGradientFunction(bceGrad, prd, tr);
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
			int n = ch.length; // ensure 1d array.
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
	// example
//	public static void main(String[]args) throws Exception
//	{
//		BCE b=new BCE();
//		// Forward pass
//		float[][] pred = {{5.2f, 10.8f},{0.79f,0.3f}};
//		float[][]trueLabel = {{1.0f, 0.0f},{1,0}};
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
//		Base pr=NDArray.wrap(flatten(pred), 2, 2).setRequiresGradient(true);
//		Base tr=new Base(flatten(trueLabel), 2, 2);
//
//		GradientDescent gd=new GradientDescent(pr);
//		float loss=10;
//		while (loss >= 0.001f)
//		{
//			Base rs=b.forward(pr, tr);
//			// print("loss " + rs);
//			loss = rs.get(0);
//			// print("---- grad ----");
//			rs.fillGrad(1);
//			rs.backward();
//			// pr.detachGradient().printArray();
//
//			gd.step();
//			gd.zeroGrad();
//			print(loss);
//			// Thread.sleep(50);
//			if (loss <= 0.0001f)
//				break;
//		}
//		print(line(20));
//		print("loss :" + loss);
//		// Test1.test(Arrays.equals(rs.base.data.getData(), loss), "loss equals with Data");
//		// Test1.test(Arrays.equals(pr.base.data.getGrads(), gradient), "gradient equals with Data");
//	}
	static void print(Object o)
	{
		System.out.println(o);
	}
}
