package gss.optimizers;

import gss.*;
import gss.arr.*;
import java.util.*;

import static gss.Util.*;

public class Adam extends Optimizer
{
    // Hyperparameters
    private final float beta1;    // Exponential decay rate for 1st moment (e.g., 0.9)
    private final float beta2;    // Exponential decay rate for 2nd moment (e.g., 0.999)
    private final float epsilon;  // Small constant for numerical stability (e.g., 1e-8)

    // Internal states (moving averages)
    private Base[] mm;  // First moment vector (mean)
    private Base[] vv;  // Second moment vector (uncentered variance)
    private int t;      // Timestep counter

	// learningRate = 1e-8f;

    // Constructor with hyperparameters
    public Adam(List<Base> prms, float lr, float beta1, float beta2, float epsilon)
	{
		this.params.clear();
        this.params.addAll(prms);
		this.learningRate = lr;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.mm = null;
        this.vv = null;
        this.t = 0;
    }
	public Adam(Base...prms, float lr, float beta1, float beta2, float epsilon)
	{
		this(Arrays.asList(prms), lr, beta1, beta2, epsilon);
	}
    // Default constructor with common hyperparameters
    public Adam(List<Base>...prms)
	{
        this(new ArrayList<>(), 0.001f, 0.8f, 0.999f, 1e-3f); // 0.001
		for (List<Base>n:prms)
			params.addAll(n);
    }
	public Adam(Base...prms)
	{
		this(Arrays.asList(prms), 0.001f, 0.8f, 0.999f, 1e-3f); // 0.001
	}
	@Override
	public void update()
	{
		if (mm == null || vv == null)
		{
            mm = new Base[params.size()]; // already 0.0f
            vv = new Base[params.size()]; // already 0.0f;
		}
		t++;
		for (int i=0;i < params.size();i++)
		{
			Base m=mm[i];
			Base v=vv[i];
			Base prm=params.get(i);
			// Data grd=ar.base.data.getGrads();
			// Data dat=ar.base.data.getData();
			if (m == null || v == null)
			{
				mm[i] = new Base(prm.shape);
				vv[i] = new Base(prm.shape);
				m = mm[i];
				v = vv[i];
			}
			update(prm, m, v);
		}
	}
    // Update parameters using Adam algorithm
    public Base update(Base prm, Base m, Base v)
	{
        // Precompute bias correction terms
        float beta1_t = (float) Math.pow(beta1, t);
        float beta2_t = (float) Math.pow(beta2, t);
        float mBiasCorrection = 1.0f / (1.0f - beta1_t);
        float vBiasCorrection = 1.0f / (1.0f - beta2_t);

        // Update each parameter
		int[] tmpShp=new int[prm.shape.length];
        for (int i = 0; i < prm.length; i++)
		{
			indexToShape(i, prm.shape, tmpShp);
			int ind=shapeToIndex(tmpShp, prm.shape, prm.strides);
			// only used for parametets.
            // Update moving averages
			float mi=m.getRaw(i);
			float vi=v.getRaw(i);
			float grd=prm.getRawGrad(ind);
            mi = beta1 * mi + (1.0f - beta1) * grd;
            vi = beta2 * vi + (1.0f - beta2) * grd * grd;

            // Compute bias-corrected estimates
            float mHat = mi * mBiasCorrection;
            float vHat = vi * vBiasCorrection;

            // Update parameter: data = data - lr * mHat / (sqrt(vHat) + epsilon)
			prm.setRaw(ind, prm.getRaw(ind) - learningRate * mHat / ((float) Math.sqrt(vHat) + epsilon));
			m.setRaw(i, mi);
			v.setRaw(i, vi);
        }
        return prm;
    }
//	public static void main(String[]args)
//	{
//		// Sample data and gradient
//		float[] data = {1.0f, 2.0f, 3.0f};  // Parameters to update
//		float[] grad = {0.5f, -0.3f, 1.2f};  // Gradients
//
//		Base ar=new Base(data).setRequiresGradient(true);
//		// set gradient to ar.
//		for (int i=0;i < grad.length;i++)
//			ar.setGrad(Util.ar(i), grad[i]); // not recomended.
//
//		Adam adam = new Adam(ar);
//		adam.learningRate = 0.01f;
//		// Perform Adam update
//		ar.printArray();
//		for (int i=0;i < 10;i++)
//		{
//			adam.update();
//			ar.printArray();
//		}
//	}
}
