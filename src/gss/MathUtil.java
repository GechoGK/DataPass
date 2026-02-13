package gss;

import gss.arr.*;
import java.util.*;

import static gss.Util.*;

public class MathUtil
{
	public static float dot(float[]a, float[]b)
	{
		if (a.length != b.length)
			error("both a and b array are not equal in length(" + (a.length + " ≠ " + b.length) + ")");
		float s=0;
		for (int i=0;i < a.length;i++)
			s += a[i] * b[i];
		return s;
	}
	// use this function if you know the output shape of the dot product of two arrays.
	// this method returns the flat array.
	public static float[]dot2(float[][]a, float[][]b)
	{
		if (a[0].length != b.length)
			error("invalid dimension for dot product: a.column(" + a[0].length + ") not equal to b.row(" + b.length + ")");
		float[]out=new float[a.length * b[0].length];
		int pos=0;
		for (int ar=0;ar < a.length;ar++)
			for (int bc=0;bc < b[0].length;bc++)
			{
				float sm=0;	
				for (int c=0;c < a[0].length;c++)
				{
					sm += a[ar][c] * b[bc][c];
				}
				out[pos++] = sm;
			}
		return out;
	}
	public static float[][]dot(float[][]a, float[][]b)
	{
		if (a[0].length != b.length)
			error("invalid dimension for dot product: a.column(" + a[0].length + ") not equal to b.row(" + b.length + ")");
		float[][]out=new float[a.length][b[0].length];
		for (int ar=0;ar < a.length;ar++)
			for (int bc=0;bc < b[0].length;bc++)
			{
				float sm=0;	
				for (int c=0;c < a[0].length;c++)
				{
					sm += a[ar][c] * b[c][bc];
				}
				out[ar][bc] = sm;
			}
		return out;
	}
	public static float[]conv1d(Base in, Base kern)
	{
		return conv1d(in, kern, null);
	}
	public static float[]conv1d(Base in, Base kern, float[]out)
	{
		// good optimization technique.
		if (in.getDim() != 1 || kern.getDim() != 1)
			throw new RuntimeException("convolve 1d error : expected 1d kernel array and 1d data.");

		int klen=kern.length;
		int len=in.length - klen + 1;
		if (out == null)
			out = new float[len];
		if (out.length != len)
			error("the parameter out array length should be " + len + ", found :" + out.length);
		float[]kcache=new float[klen];
		float[]incache=new float[in.length];
		// caching the kernel, and the portion of data.
		float sm=0;
		int p=0;
		for (int k=klen - 1;k >= 0;k--)
		{
			float kv=kern.get(k);
			float iv=in.get(p);
			kcache[p] = kv;
			incache[p] = iv;
			sm += iv * kv;
			p++;
		}
		out[0] += sm;
		// kernel cached, use cached kernel, to gain some performance.
		for (int i=1;i < len;i++) // starts from 1, 0 is used for kernel caching...
		{
			int k=klen - 1;
			float iv=in.get(i + k);
			incache[i + k] = iv;
			sm = iv * kcache[k];
			k--;
			for (;k >= 0;k--)
			{
				sm += incache[i + k] * kcache[k];
			}
			out[i] += sm;
		}
		return out;
	}
	public static float[][] conv2d(Base inp, Base krn)
	{
		return conv2d(inp, krn, null, null);
	}
	public static float[][] conv2d(Base inp, Base krn, Base bs, float[][]out)
	{
		if (krn.getDim() != 2 || krn.shape[0] != krn.shape[1])
			error("the kernel dimension should be 2: found = " + krn.getDim());
		if (inp.getDim() != 2)
			error("input dimension needs to be 2, found :" + inp.getDim());

		int k_size=krn.shape[0];
		int outputR=(inp.shape[0] - krn.shape[0]) + 1;
		int outputC=(inp.shape[1] - k_size) + 1;

		if (out == null)
			out = new float[outputR][outputC];
		if (out.length != outputR || out[0].length != outputC)
			error("the parameter out array length should be (" + (outputR + ", " + outputC) + "), found :(" + out.length + ", " + out[0].length + ")");

		boolean hasB=bs != null;

		float[][] in=copy2(inp);
		float[][] kern=copy2(krn);

		for (int or=0;or < outputR;or++)
		{
			for (int oc=0;oc < outputC;oc++) // we used one loop for caching kernel.
			{
				float sm = 0;
				for (int kr=k_size - 1,ir=0;kr >= 0;kr--,ir++)
				{
					for (int kc=k_size - 1,ic=0;kc >= 0;kc--,ic++)
					{
						sm += in[ir + or][ic + oc] * kern[kr][kc];
					}
				}
				if (!hasB)
					out[or][oc] = sm;
				else
				{
					out[or][oc] = sm + bs.get(or, oc);
				}
			}
		}
		return out;
	}
	public static float[] maxPool1d(Base in, int poolSize)
	{
		return maxPool1d(in, poolSize, null, null);
	}
	public static float[] maxPool1d(Base in, int poolSize, float[]out, int[]indexOut)
	{
		if (in.getDim() != 1)
			error("for 1d pool, the input data should be 1d array.");
		if (in.shape[0] % poolSize != 0)
			error("the input size should be divisble by pool size");
		if (out == null)
			out = new float[in.shape[0] / poolSize];
		if (out.length != in.shape[0] / poolSize)
			error("the output array size should be (" + in.shape[0] / poolSize + "), found :" + out.length);
		for (int i=0,pp=0;i < in.shape[0];i += poolSize,pp++)
		{
			float cp=in.get(i);
			int idx=i;
			for (int p=1;p < poolSize;p++)
			{
				int ip=i + p;
				float v=in.get(ip);
				if (v > cp)
				{
					cp = v;
					idx = ip;
				}
			}
			out[pp] = cp;
			if (indexOut != null)
				indexOut[pp] = idx;
		}
		return out;
	}
	public static float[] averagePool1d(Base in, int poolSize)
	{
		return averagePool1d(in, poolSize, null);
	}
	public static float[] averagePool1d(Base in, int poolSize, float[]out)
	{
		if (in.getDim() != 1)
			error("for 1d pool, the input data should be 1d array.");
		if (in.shape[0] % poolSize != 0)
			error("the input size should be divisble by pool size");
		if (out == null)
			out = new float[in.shape[0] / poolSize];
		if (out.length != in.shape[0] / poolSize)
			error("the output array size should be (" + in.shape[0] / poolSize + "), found :" + out.length);
		for (int i=0,pp=0;i < in.shape[0];i += poolSize,pp++)
		{
			float sm=0;
			for (int p=0;p < poolSize;p++)
			{
				sm += in.get(i + p);
			}
			out[pp] = sm / poolSize;
		}
		return out;
	}
	public static float[][]maxPool2d(Base in, int...poolSize)
	{
		return maxPool2d(in, null, null, poolSize);
	}
	public static float[][]maxPool2d(Base in, float[][]out, int[][][]indexOut, int...poolSize)
	{
		// set to index array.
		if (poolSize.length != 2)
			error("2d array pool size expected.");
		if (in.getDim() != 2)
			error("2d input array expected. found :" + in.getDim());
		if (in.shape[0] % poolSize[0] != 0 || in.shape[1] % poolSize[1] != 0)
			error("the input size should be divisible by pool size. in" + Arrays.toString(in.shape) + ", poolSize" + Arrays.toString(poolSize));
		int[]outShape={in.shape[0] / poolSize[0],in.shape[1] / poolSize[1]};
		if (out == null)
			out = new float[outShape[0]][outShape[1]];
		if (out.length != outShape[0] || out[0].length != outShape[1])
			error("the output array size should be (" + outShape[0] / poolSize[0] + ", " + outShape[1] / poolSize[1] + "), found :(" + out.length + ", " + out[0].length + ")");
		for (int ir=0,or=0;ir < in.shape[0];ir += poolSize[0],or++)
			for (int ic=0,oc=0;ic < in.shape[1];ic += poolSize[1],oc++)
			{
				float mx=in.get(ir, ic);
				int[] idx={ir,ic};
				for (int r=0;r < poolSize[0];r++)
					for (int c=0;c < poolSize[1];c++)
					{
						float v=in.get(ir + r, ic + c);
						if (v > mx)
						{
							mx = v;
							idx[0] = ir + r;
							idx[1] = ic + c;
						}
					}
				out[or][oc] = mx;
				if (indexOut != null)
					indexOut[or][oc] = idx;
			}
		return out;
	}
	public static float[][] averagePool2d(Base in, int...poolSize)
	{
		return averagePool2d(in, null, poolSize);
	}
	public static float[][] averagePool2d(Base in, float[][]out, int...poolSize)
	{
		if (poolSize.length != 2)
			error("2d array pool size expected.");
		if (in.getDim() != 2)
			error("2d input array expected. found :" + in.getDim());
		if (in.shape[0] % poolSize[0] != 0 || in.shape[1] % poolSize[1] != 0)
			error("the input size should be divisible by pool size. in" + Arrays.toString(in.shape) + ", poolSize" + Arrays.toString(poolSize));
		int[]outShape={in.shape[0] / poolSize[0],in.shape[1] / poolSize[1]};
		if (out == null)
			out = new float[outShape[0]][outShape[1]];
		if (out.length != outShape[0] || out[0].length != outShape[1])
			error("the output array size should be (" + outShape[0] / poolSize[0] + ", " + outShape[1] / poolSize[1] + "), found :(" + out.length + ", " + out[0].length + ")");
		int len=poolSize[0]   * poolSize[1];
		for (int ir=0,or=0;ir < in.shape[0];ir += poolSize[0],or++)
			for (int ic=0,oc=0;ic < in.shape[1];ic += poolSize[1],oc++)
			{
				float sm=0;
				for (int r=0;r < poolSize[0];r++)
					for (int c=0;c < poolSize[1];c++)
					{
						sm += in.get(ir + r, ic + c);
					}
				out[or][oc] = sm / len;
			}
		return out;
	}
	public static float[] variance()
	{
		return null;
	}
	public static float[]sum()
	{
		return null;
	}
	public static float[]copy1(Base in)
	{
		if (in.getDim() != 1)
			error("the input dimension must be 1, found :" + in.getDim());
		float[] o=new float[in.shape[0]];
		for (int or=0;or < in.shape[0];or++)
			o[or] = in.get(or);
		return o;
	}
	public static float[][]copy2(Base in)
	{
		if (in.getDim() != 2)
			error("the input dimension must be 2, found :" + in.getDim());
		float[][] o=new float[in.shape[0]][in.shape[1]];
		for (int or=0;or < in.shape[0];or++)
			for (int oc=0;oc < in.shape[1];oc++) // we used one loop for caching kernel.
				o[or][oc] = in.get(or, oc);
		return o;
	}
	public static float[][][]copy3(Base in)
	{
		if (in.getDim() != 3)
			error("the input dimension must be 3, found :" + in.getDim());
		float[][][] o=new float[in.shape[0]][in.shape[1]][in.shape[2]];
		for (int od=0;od < in.shape[0];od++)
			for (int or=0;or < in.shape[1];or++)
				for (int oc=0;oc < in.shape[2];oc++)
					o[od][or][oc] = in.get(od, or, oc);
		return o;
	}
	public static float[][][][]copy4(Base in)
	{
		if (in.getDim() != 4)
			error("the input dimension must be 4, found :" + in.getDim());
		float[][][][] o=new float[in.shape[0]][in.shape[1]][in.shape[2]][in.shape[3]];
		for (int ob=0;ob < in.shape[0];ob++)
			for (int od=0;od < in.shape[1];od++)
				for (int or=0;or < in.shape[2];or++)
					for (int oc=0;oc < in.shape[3];oc++)
						o[ob][od][or][oc] = in.get(ob, od, or, oc);
		return o;
	}
	// set float array value to NDArray.
	public static Base set(Base arr, float[]val)
	{
		if (arr.getDim() != 1 || arr.length != val.length)
			error(arr.getDim() != 1 ?"input dimension doesn't match to input value:(" + arr.getDim() + "≠" + 1 + ").": "arr.length doesn't match with value length:(" + arr.length + "≠" + val.length + ").");
		for (int i=0;i < arr.shape[0];i++)
			arr.set(ar(i), val[i]);
		return arr;
	}
	public static Base set(Base arr, float[][]val)
	{
		if (arr.getDim() != 2 || arr.length != val.length * val[0].length)
			error(arr.getDim() != 2 ?"input dimension doesn't match to input value:(" + arr.getDim() + "≠" + 2 + ").": "arr.length doesn't match with value length:(" + arr.shape[0] + "," + arr.shape[1] + " ≠ " + val.length + "," + val[0].length + ").");
		for (int i=0;i < arr.shape[0];i++)
			for (int j=0;j < arr.shape[1];j++)
				arr.set(ar(i, j), val[i][j]);
		return arr;
	}
	public static Base set(Base arr, float[][][]val)
	{
		if (arr.getDim() != 3 || arr.length != val.length * val[0].length * val[0][0].length)
			error(arr.getDim() != 3 ?"input dimension doesn't match to input value:(" + arr.getDim() + "≠" + 3 + ").": "arr.length doesn't match with value length:(" + arr.shape[0] + "," + arr.shape[1] + "," + arr.shape[2] + " ≠ " + val.length + "," + val[0].length + "," + val[0][0].length + ").");
		for (int d=0;d < arr.shape[0];d++)
			for (int r=0;r < arr.shape[1];r++)
				for (int c=0;c < arr.shape[2];c++)
					arr.set(ar(d, r, c), val[d][r][c]);
		return arr;
	}
	public static Base set(Base arr, float[][][][]val)
	{
		if (arr.getDim() != 4 || arr.length != val.length * val[0].length * val[0][0].length * val[0][0][0].length)
			error(arr.getDim() != 4 ?"input dimension doesn't match to input value:(" + arr.getDim() + "≠" + 4 + ").": "arr.length doesn't match with value length:(" + arr.shape[0] + "," + arr.shape[1] + "," + arr.shape[2] + "," + arr.shape[3] + " ≠ " + val.length + "," + val[0].length + "," + val[0][0].length + "," + val[0][0][0].length + ").");
		for (int b=0;b < arr.shape[0];b++)
			for (int d=0;d < arr.shape[1];d++)
				for (int r=0;r < arr.shape[2];r++)
					for (int c=0;c < arr.shape[3];c++)
						arr.set(ar(b, d, r, c), val[b][d][r][c]);
		return arr;
	}
	// append values.
	public static Base append(Base arr, float[]val)
	{
		if (arr.getDim() != 1 || arr.length != val.length)
			error(arr.getDim() != 1 ?"input dimension doesn't match to input value:(" + arr.getDim() + "≠" + 1 + ").": "arr.length doesn't match with value length:(" + arr.length + "≠" + val.length + ").");
		for (int i=0;i < arr.shape[0];i++)
			arr.append(ar(i), val[i]);
		return arr;
	}
	public static Base append(Base arr, float[][]val)
	{
		if (arr.getDim() != 2 || arr.length != val.length * val[0].length)
			error(arr.getDim() != 2 ?"input dimension doesn't match to input value:(" + arr.getDim() + "≠" + 2 + ").": "arr.length doesn't match with value length:(" + arr.shape[0] + "," + arr.shape[1] + " ≠ " + val.length + "," + val[0].length + ").");
		for (int i=0;i < arr.shape[0];i++)
			for (int j=0;j < arr.shape[1];j++)
				arr.append(ar(i, j), val[i][j]);
		return arr;
	}
	public static Base append(Base arr, float[][][]val)
	{
		if (arr.getDim() != 3 || arr.length != val.length * val[0].length * val[0][0].length)
			error(arr.getDim() != 3 ?"input dimension doesn't match to input value:(" + arr.getDim() + "≠" + 3 + ").": "arr.length doesn't match with value length:(" + arr.shape[0] + "," + arr.shape[1] + "," + arr.shape[2] + " ≠ " + val.length + "," + val[0].length + "," + val[0][0].length + ").");
		for (int d=0;d < arr.shape[0];d++)
			for (int r=0;r < arr.shape[1];r++)
				for (int c=0;c < arr.shape[2];c++)
					arr.append(ar(d, r, c), val[d][r][c]);
		return arr;
	}
	public static Base append(Base arr, float[][][][]val)
	{
		if (arr.getDim() != 4 || arr.length != val.length * val[0].length * val[0][0].length * val[0][0][0].length)
			error(arr.getDim() != 4 ?"input dimension doesn't match to input value:(" + arr.getDim() + "≠" + 4 + ").": "arr.length doesn't match with value length:(" + arr.shape[0] + "," + arr.shape[1] + "," + arr.shape[2] + "," + arr.shape[3] + " ≠ " + val.length + "," + val[0].length + "," + val[0][0].length + "," + val[0][0][0].length + ").");
		for (int b=0;b < arr.shape[0];b++)
			for (int d=0;d < arr.shape[1];d++)
				for (int r=0;r < arr.shape[2];r++)
					for (int c=0;c < arr.shape[3];c++)
						arr.append(ar(b, d, r, c), val[b][d][r][c]);
		return arr;
	}
}
