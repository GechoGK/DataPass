package gss.layers;

import gss.*;
import gss.arr.*;
import gss.layers.*;
import java.util.*;

public class Dropout extends Module
{
	private float drp;
    private Random random;
    private boolean isTraining;
    private int[] mask; // Store the dropout mask for backward pass (if needed)

    public Dropout(float p)
	{
        this.drp = p;
        this.random = new Random();
        this.isTraining = true;
    }
	@Override
	public Base forward(Base input)
	{
		if (!isTraining || !input.hasGradient())
			return input;
		Base in=input.as1DArray();
		float[] output = new float[in.length];
        mask = new int[output.length]; // Store which neurons are active, it helps for backpropagation.
        float scale = 1.0f / (1.0f - drp);
        for (int i = 0; i < output.length; i++)
		{
            if (random.nextFloat() < drp)
			{
                output[i] = 0.0f;
                mask[i] = 0; // Neuron dropped
            }
			else
			{
                output[i] = in.get(i) * scale;
                mask[i] = 1; // Neuron active
            }
        }
        Base b = new Base(output, input.shape);
		b.setRequiresGradient(in.hasGradient());
		b.setGradientFunctionS(GradFunc.maskGradient, mask, in);
		// b.setGradientParams(mask);
		return b;
    }
//	public static GradFunc dropoutGradient = new GradFunc("dropout"){
//		@Override
//		public Base backward(Base host, Base[] childs, Object params)
//		{
//			Base grd=host.detachGradient().as1DArray();// .base.data.getGrads();
//			int[] mask=((int[])params);
//			Base c1=childs[0];
//			if (host.length != c1.length)
//				throw new RuntimeException("unable to compute dropout backpropagation.: invalid array length between the host and the child.");
//			Base rs=NDArray.mul(grd, new Base(Util.asFloat(mask)));
//			c1.setGrad(rs);
//			return null;
//		}
//	};
}
