package gss.layers;

import gss.*;
import gss.arr.*;

public class BatchNorm extends Module
{
	private Base gamma,beta;
	private float eps=0;

	public BatchNorm(int batch_size)
	{
		gamma = newParam(NDArray.ones(batch_size, 1));
		beta = newParam(NDArray.zeros(batch_size, 1));
		this.eps = 1e-5f;
	}
	@Override
	public Base forward(Base dataIn)
	{
		// gradient block start
		Base in = dataIn.reshape(dataIn.shape[0], -1);
		Base mean = NDArray.mean(in, 0).reshape(1, in.shape[1]);

		Base var = NDArray.variance(in, 0).reshape(mean.shape);
		Base norm = NDArray.div(NDArray.sub(in , mean), NDArray.sqrt(NDArray.add(var, eps)));

		Base out = NDArray.add(NDArray.mul(gamma, norm), beta);
		out = out.reshape(dataIn.shape);
		// gradient block end.
		return out;
	}
}
