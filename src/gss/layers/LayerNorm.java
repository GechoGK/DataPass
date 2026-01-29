package gss.layers;

import gss.*;
import gss.arr.*;

public class LayerNorm extends Module
{
	private Base gamma,beta;
	private float eps=0;

	public LayerNorm(int features_shape)
	{
		gamma = newParam(NDArray.ones(features_shape));
		beta = newParam(NDArray.zeros(features_shape));
		this.eps = 1e-5f;
	}
	@Override
	public Base forward(Base dataIn)
	{
		return forward(dataIn, new int[0]);
	}
	public Base forward(Base dataIn, int[]axis)
	{
		// gradient block start
		Base in = dataIn.as2DArray();
		Base mean = NDArray.mean(in, axis).reshape(in.shape[0], 1);
		Base var = NDArray.variance(in, axis).reshape(mean.shape);

		Base norm = NDArray.div(NDArray.sub(in , mean), NDArray.sqrt(NDArray.add(var, eps)));

		Base out = NDArray.add(NDArray.mul(gamma, norm), beta);

		out = out.reshape(dataIn.shape);

		// gradient block end.
		return out;
	}
}
