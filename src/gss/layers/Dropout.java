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
		if (!isTraining)
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
        return new Base(output, input.shape);
    }
//	public static GradFunc DropoutGradient =new GradFunc("dropout"){
//		@Override
//		public NDArray backward(NDArray host, NDArray[] childs, Object[] params)
//		{
//			float[] grd=host.base.data.getGrads();
//			int[] mask=(int[])params[0];
//			float[] ngrd=Dropout.backward(grd, mask);
//			NDArray ch1=childs[0];
//			ch1.base.data.setGrad(ngrd); // don't use this method for chain rule. because it doesn't append the value, instead it replaces it.
//			return null;
//		}
//	};
//    // Hypothetical backward method (gradient propagation)
//    public static float[] backward(float[] grad, int[] mask)
//	{
//        // During backward pass: propagate gradients only through active neurons
//        float[] gradInput = new float[grad.length];
//        for (int i = 0; i < grad.length; i++)
//		{
//            gradInput[i] = mask[i] * grad[i];
//        }
//        return gradInput;
//    }
//
//    public void setTraining(boolean isTraining)
//	{
//        this.isTraining = isTraining;
//    }
}
