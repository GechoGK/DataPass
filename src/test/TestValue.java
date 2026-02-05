package test;

import gss.*;
import gss.arr.*;

import static gss.MathUtil.*;
import static gss.Util.*;

public class TestValue
{
	public static Base conv2d(Base input, Base kernels, Base biase)
	{
		if (input.shape[1] != kernels.shape[1])
			error("unable to compute conv2d invalid featire size.");
		Base in=input.as4DArray(); // 4d array (batch_size, num_channels, input_size, input_size);
		int numKernels=kernels.shape[0];
		int numChannels=kernels.shape[1];
		int outputSize=kernels.shape[2];
		boolean haveBiase=biase != null;

		int k_size=kernels.shape[3];
		int outputR=(input.shape[3] - kernels.shape[3]) + 1;
		int outputC=(input.shape[3] - k_size) + 1;

		// int[] shape={in.shape[1],in.shape[0],numKernels,outputSize,outputSize};
		// Base out=new Base(shape).setRequiresGradient(in.hasGradient());
		int[] shape={in.shape[0],numKernels,outputSize,outputSize};
		Value[][][][][]out=new Value[in.shape[1]][in.shape[0]][numKernels][outputSize][ outputSize];
		// float[][][][] out=new float[in.shape[0]][numKernels][outputSize][outputSize];
		// output = batch,numKernels,outSize,outSize

		for (int bs=0;bs < in.shape[0];bs++) // loop over batch size
		{
			for (int ks=0;ks < numKernels;ks++) // loop over number of kernels
			{
				for (int cs=0;cs < in.shape[1];cs++) // loop over channels size
				{
					for (int or=0;or < outputR;or++)
					{
						for (int oc=0;oc < outputC;oc++) // we used one loop for caching kernel.
						{
							Value sm = null;
							for (int kr=k_size - 1,ir=0;kr >= 0;kr--,ir++)
							{
								for (int kc=k_size - 1,ic=0;kc >= 0;kc--,ic++)
								{
									Value v1=input.getValue(bs, cs, ir + or, ic + oc);
									Value v2=kernels.getValue(ks, cs, kr, kc);
									//sm += v1 * v2;
									sm = sm == null ?v1.mul(v2): sm.add(v1.mul(v2));
								}
							}
							// out[or][oc] += sm;
							if (!haveBiase)
							 	out[bs][ks][cs][or][oc] = sm;
							else
							 	out[bs][ks][cs][or][oc] = sm.add(biase.getValue(ks, or, oc));
							// out.setValue(sm, cs, bs, ks, or, oc);
						}
					}
				}
			}
		}
		Value[][][][]out2=sum(out);
		Base b=fromValue(out2, shape);
		b.setRequiresGradient(in.hasGradient() || kernels.hasGradient());
		b.setGradientFunctionS(GradFunc.itemGradient);
		return b;
		// out.setGradientFunctionS(GradFunc.itemGradient);
		// out = NDArray.sum(out, 0);
		// return out;
	}
	public static Value[][][][]sum(Value[][][][][] arr)
	{
		Value[][][][]o=arr[0];
		for (int c=1;c < arr.length;c++)
		{
			for (int i=0;i < arr[c].length;i++)
			{
				for (int j=0;j < arr[c][i].length;j++)
				{
					for (int k=0;k < arr[c][i][j].length;k++)
					{
						for (int l=0;l < arr[c][i][j][k].length;l++)
						{
							o[i][j][k][l] = o[i][j][k][l].add(arr[c][i][j][k][l]);
						}
					}
				}
			}
		}
		return o;
	}
	public static Base fromValue(Value[][][][] arr, int...shp)
	{
		Base b=new Base(length(shp));
		int p=0;
		for (int i=0;i < arr.length;i++)
		{
			for (int j=0;j < arr[i].length;j++)
			{
				for (int k=0;k < arr[i][j].length;k++)
				{
					for (int l=0;l < arr[i][j][k].length;l++)
					{
						b.setValue(arr[i][j][k][l], p++);
					}
				}
			}
		}
		return b.reshapeLocal(shp);
	}
	public static Base relu2(Base arr)
	{
		// Value[] f=arr.base.toValueArray();
		Base out=new Base(arr.shape).setRequiresGradient(arr.hasGradient());
		out.setGradientFunction(GradFunc.itemGradient);
		for (int i=0;i < arr.length;i++)
		{
			Value v=arr.getValue(arr.indexToShape(i));
			v = v.getData() > 0 ?v.step(): new Value(0);
			out.setValue(v, out.indexToShape(i));
		}
		return out;
	}
	public static Base sigmoid2(Base arr)
	{
		/*
		 float sigmoidForward(float x) {
		 return 1.0f / (1.0f + (float) Math.exp(-x));
		 }
		 */
		// Value[] f=arr.base.toValueArray();
		Base out=new Base(arr.shape).setRequiresGradient(arr.hasGradient());
		out.setGradientFunction(GradFunc.itemGradient);
		for (int i=0;i < arr.length;i++)
		{
			Value v=arr.getValue(arr.indexToShape(i));
			v = new Value(1).div(new Value(1).add(v.mul(new Value(-1)).exp()));
			out.setValue(v, out.indexToShape(i));
		}
		return out;
	}
	public static Base tanh2(Base arr)
	{
		// Value[] f=arr.base.toValueArray();
		Base out=new Base(arr.shape).setRequiresGradient(arr.hasGradient());
		out.setGradientFunction(GradFunc.itemGradient);
		for (int i=0;i < arr.length;i++)
		{
			Value v=arr.getValue(arr.indexToShape(i)).tanh();
			out.setValue(v, out.indexToShape(i));
		}
		return out;
	}
	public static Base dot2(Base a, Base b)
	{
		int[] out=NDArray.dotShape(a.shape, b.shape);
		a = a.as2DArray();
		if (b.getDim() < 2)
			b = b.reshape(1, -1);
		else if (b.getDim() == 2)
		{
			b = b.transpose(1, 0);
		}
		else if (b.getDim() > 2)
		{
			b = b.transpose(NDArray.dotAxis(b.getDim()));
			b = b.reshape(-1, b.shape[b.shape.length - 1]);
		}
		if (b.shape[b.shape.length - 1] != a.shape[a.shape.length - 1])
			throw new RuntimeException("invalid shape dor dot product.");
		int[] sh={a.shape[0],b.shape[0]};
		Base bs=new Base(sh);
		for (int ar=0;ar < a.shape[0];ar++)
			for (int br=0;br < b.shape[0];br++)
			{
				Value sm=new Value(0);	
				for (int c=0;c < a.shape[1];c++)
				{
					sm = sm.add(a.getValue(ar, c).mul(b.getValue(br, c)));
				}
				bs.setValue(sm, Util.ar(ar, br));
			}
		bs.setRequiresGradient(a.hasGradient() && b.hasGradient());
		bs = bs.setGradientFunction(GradFunc.dotGradient, a, b).reshape(out);
		return bs;
	}
	public static Base convolve1d2(Base a, Base b, Base out)
	{
		// valid convolution.
		if (a.getDim() != 1 || b.getDim() != 1)
			throw new RuntimeException("convolve 1d error : expected 1d kernel array and 1d data.");
		int len=Math.max(a.length, b.length) - Math.min(a.length, b.length) + 1;
		if (out == null)
			out = new Base(len).setRequiresGradient(a.hasGradient() && b.hasGradient());
		int wi=0;
		int iinc=a.length > b.length ?1: 0;
		int kinc=b.length > a.length ?1: 0;
		int wr=b.length > a.length ?len - 1: 0;
		for (int w=0;w < len;w++)
		{
			Value sm=new Value(0);
			int kl=b.length - 1;
			for (int i=0;i < Math.min(a.length, b.length);i++)
			{
				Value aa=b.getValue(kl - wr);
				Value bb=a.getValue(i + wi);
				sm = sm.add(aa.mul(bb));
				kl--;
			}
			wr -= kinc;
			wi += iinc;
			out.setValue(sm, w);
		}
		return out.setGradientFunction(GradFunc.itemGradient);
	}
	public static Base bce2(Base prd, Base tr)
	{
		// doesn't work.
		Base b=new Base(1).setRequiresGradient(prd.hasGradient());
		int n = prd.length;
		Value loss = new Value(0);
		Value epsilon = new Value(1e-7f); // Avoid log(0)

		for (int i = 0; i < n; i++)
		{
			// Clip predictions to [epsilon, 1-epsilon]
			Value tarF=tr.getValue(i);
			Value p = Value.max(epsilon, Value.min(prd.getValue(i), new Value(1).sub(epsilon)));
			// loss = loss.add(p);
			loss = loss.add(tarF.mul(p.log()).add((new Value(1).sub(tarF).mul(new Value(1).sub(p).log()))));
		}

		// Average and negate (BCE formula)
		loss = loss.mul(new Value(-1f)).div(new Value(n));
		b.setValue(loss, 0);
		b.setGradientFunction(GradFunc.itemGradient);
		return b;
	}
	public static Base mse2(Base prd, Base tr)
	{
		int n = prd.length;
		Value loss = new Value(0.0f);
		Base b=new Base(1).setRequiresGradient(prd.hasGradient());
		// Compute sum of squared errors
		for (int i = 0; i < n; i++)
		{
			Value diff = prd.getValue(i).sub(tr.getValue(i));
			loss = loss.add(diff.pow(new Value(2)));
		}

		// Average the sum
		loss = loss.div(new Value(n)); // /= n;
		b.setValue(loss, 0);
		b.setGradientFunction(GradFunc.itemGradient);
		return b;
	}
}
