package gss.optimizers;

import gss.*;
import gss.arr.*;
import java.util.*;

import static gss.Util.*;

public class SGDM extends Optimizer
{
	// this optimizer expects the parameters shape are not modified(not transposed, broadcasted, ...)

    private float momentum;
    private Base[] velocity;  // Tracks accumulated velocity for momentum

    // Constructor: Initialize learning rate and momentum factor
    public SGDM(List<Base> prms, float learningRate, float momentum)
	{
        super(learningRate);
		this.params.addAll(prms);
        this.momentum = momentum;
        this.velocity = null;  // Velocity initialized lazily on first `forward` call
    }
	public SGDM(List<Base>...prms)
	{
		this(new ArrayList<>(), 0.01f, 0.85f);
		for (List<Base> p:prms)
			params.addAll(p);
	}
	public SGDM(Base...prms, float lr,  float momtm)
	{
		this(Arrays.asList(prms), lr, momtm);
	}
	public SGDM(Base...prms)
	{
		this(Arrays.asList(prms), 0.01f, 0.85f);
	}
	@Override
	public void step()
	{
		if (velocity == null)
			velocity = new Base[params.size()];
		for (int p=0;p < params.size();p++)
		{
			Base prm=params.get(p);
			Base vl=velocity[p];
			if (vl == null)
			{
				velocity[p] = new Base(prm.shape);
				vl = velocity[p];
			}
			// update(vl, grad, data);
			int[] tmpShp=new int[prm.shape.length];
			for (int i = 0; i < prm.length; i++)
			{
				indexToShape(i, prm.shape, tmpShp);
				int ind=shapeToIndex(tmpShp, prm.shape, prm.strides);

				float veli=vl.getRaw(i);
				veli = momentum * veli + prm.getRawGrad(ind);
				prm.setRaw(ind, prm.getRaw(ind) - learningRate * veli);
				vl.setRaw(i, veli);
			}
		}
	}
    // Update parameters using momentum
//    private float[] update(float[]vel, float[] grad, float[] data)
//	{
//        // Apply momentum update rule:
//        // velocity = momentum * velocity + grad
//        // data = data - learningRate * velocity
//        for (int i = 0; i < data.length; i++)
//		{
//            vel[i] = momentum * vel[i] + grad[i];
//            data[i] -= learningRate * vel[i];
//        }
//
//        return data;  // Updated parameters (modified in-place)
//    }
//	public static void main(String[]args)
//	{
//		// Example data (parameters) and gradient
//		float[] data = {1.0f, 2.0f, 3.0f};
//		float[] grad = {0.1f, 0.2f, 0.3f};
//
//		Base ar=new Base(data).setRequiresGradient(true);
//		for (int i=0;i < grad.length;i++)
//			ar.setGrad(Util.ar(i), grad[i]);
//
//		ar.printArray();
//		// ar.detachGradient().printArray();
//
//		// Initialize optimizer with learning rate 0.01 and momentum 0.9
//		SGDM optimizer = new SGDM(Arrays.asList(ar), 0.01f, 0.9f);
//		// Update parameters using momentum
//		optimizer.update();
//
//		ar.printArray();
//		// ar.detachGradient().printArray();
//		// System.out.println(Arrays.toString(data));  
//		// Output: [0.999, 1.998, 2.997] (after first step)
//	}
}
