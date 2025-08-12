import gss.*;
import gss.act.*;
import gss.arr.*;
import gss.layers.*;
import gss.lossfunctions.*;
import gss.optimizers.*;

import static gss.Util.*;
import static gss.arr.GradFunc.*;

public class Test2_Func
{
	public static void main2(String[] args) throws Exception
	{

		new Test2_Func().a();
		System.out.println(line(50));

	}
	void a() throws Exception
	{

		Base b1=NDArray.arange(10).setRequiresGradient(true);
		Base b2=NDArray.arange(10).setRequiresGradient(true);

		b1.printArray();
		b2.printArray();
		Base b3=new Base(10).setRequiresGradient(true);
		for (int i=0;i < b1.length;i++)
		{
			Value r=b1.getValue(i).add(b2.getValue(i));
			b3.setValue(r, i);
		}
		b3.setGradientFunction(GradFunc.itemGradient);
		print(b3);
		b3.printArray();
		print(line(30));
		b3.setGrad(2);
		b3.backward();

		b1.detachGradient().printArray();
		b2.detachGradient().printArray();
	}
	void tree(Base arr, String t)
    {
		System.out.println(t + arr);
		if (arr.gradientFunction == GradFunc.itemGradient)
        {
			System.out.println(t + "listing child gradient");
			for (int i=0;i < arr.length;i++)
				treeV(arr.getValue(i), t);
		}
		if (arr.childs != null && arr.childs.size() != 0)
			for (Base ar:arr.childs)
				tree(ar, t.replace("_", " ").replace("|", " ") + "|_____ ");
	}
	void treeV(Value vl, String t)
    {
		System.out.println(t + vl);
		if (vl.args != null && vl.args.size() != 0)
			for (Value  vv:vl.args)
				treeV(vv, t.replace("_", " ").replace("|", " ") + "|_____ ");
	}
	void test8()
	{
		Base a=NDArray.arange(3 * 5).reshapeLocal(3, 5).setRequiresGradient(true);
		Base b=NDArray.arange(5 * 2).reshapeLocal(5, 2).setRequiresGradient(true);

		Base c=NDArray.dot(a, b);

		print(a);
		a.printArray();
		print(b);
		b.printArray();
		print(c);
		c.printArray();
		c.setGrad(1);
		c.backward();
		print(line(30));
		a.detachGradient().printArray();
		print(line(10));
		b.detachGradient().printArray();

	}
	Base dot(Base a, Base b)
	{
		int[] out=dotShape(a.shape, b.shape);
		a = a.as2DArray();
		if (b.getDim() < 2)
			b = b.reshape(1, -1);
		else if (b.getDim() == 2)
		{
			b = b.transpose(1, 0);
		}
		else if (b.getDim() > 2)
		{
			b = b.transpose(dotAxis(b.getDim()));
			b = b.reshape(-1, b.shape[b.shape.length - 1]);
		}
		if (b.shape[b.shape.length - 1] != a.shape[a.shape.length - 1])
			throw new RuntimeException("invalid shape dor dot product.");
		int[] sh={a.shape[0],b.shape[0]};
		float[] outData=new float[a.shape[0] * b.shape[0]];
		for (int ar=0;ar < a.shape[0];ar++)
			for (int br=0;br < b.shape[0];br++)
			{
				float sm=0;	
				for (int c=0;c < a.shape[1];c++)
				{
					sm += a.get(ar, c) * b.get(br, c);
				}
				outData[shapeToIndex(new int[]{ar,br}, sh)] = sm;
			}
		Base bs=new Base(outData, out).setRequiresGradient(a.requiresGradient() | b.requiresGradient());
		bs.setGradientFunction(GradFunc.dotGradient, a, b);
		return bs;
	}
	int[] dotAxis(int len)
	{
		// the len value expectes to be > 2
		int[]a=new int[len];
		for (int i=0;i < len;i++)
			a[i] = i;
		int t=a[len - 1];
		a[len - 1] = a[len - 2];
		a[len - 2] = t;
		return a;
	}
	private static int[] dotShape(int[]sh1, int[]sh2)
	{
		// this method expect the array before transposed.
		if (n(sh1, 0) != n(sh2, sh2.length == 1 ?0: 1))
			throw new RuntimeException("incompatable shape for dot product.");
		int len=sh1.length + sh2.length - 2;
		// print("length :", len);
		if (len == 0)
			return new int[]{1};
		int[]ns=new int[len];
		for (int i=0;i < sh1.length - 1;i++)
			ns[i] = sh1[i];
		int sh2Start=sh1.length - 1;
		for (int i=0;i < sh2.length - 2;i++)
			ns[i + sh2Start] = sh2[i];
		if (sh2.length >= 2)
			ns[ns.length - 1] = sh2[sh2.length - 1];
		return ns;
	}
	void draw(Base bs, String s)
	{
		System.out.println(s + bs.gradientFunction);
		for (Base b:bs.childs)
			draw(b, "   " + s);
	}
	void test7()
	{
		// doesn't work.
		print(decString("Test 7.0 approximation test with different loss functions.", 7));
		/*
		 BCE doesn*t work as expected.
		 MCCE doesn't works as expected.
		 al these works without activation functions.
		 */
		// if (new Boolean(true) == true)
		// 	throw new RuntimeException("review the code first");
		int input=2;
		int output=3;
		Base w1=NDArray.rand(input, 5).setRequiresGradient(true);
		Base w2=NDArray.rand(5, output).setRequiresGradient(true);
		Base b1=NDArray.ones(5).setRequiresGradient(true);
		Base b2=NDArray.ones(output).setRequiresGradient(true);

		Base in=NDArray.wrap(new float[]{0.5f,0.2f}, input);
		Base tr=NDArray.wrap(new float[]{1,0,1,0,1,0}, 2, 3);

		Optimizer opt=new GradientDescent(w1, w2, b1, b2);

		// trainMSE(opt, w1, w2, b1, b2, in, tr); // ≈ 19755, 24330, 10205, 15488, 7940, 6515, 6515, 6817 millis
		// trainMAE(opt, w1, w2, b1, b2, in, tr); // ≈ 80709, 33369, 27102, 19464, 15508, 22978  millis
		// trainBCE(opt, w1, w2, b1, b2, in, tr); // ≈ 79235, 74982, 32427, 15759, 16387, 13459, 16758 millis
		trainMCCE(opt, w1, w2, b1, b2, in, tr); // slow and inaccurate // ≈ 79272, 20064, 22044, 30453, 7360, 7421, 5817, 7141, 4452, 5733   millis

		System.out.println("completed!");

	}
	void trainMSE(Optimizer opt, Base w1, Base w2, Base b1, Base b2, Base in, Base tr)
	{
		Base output=null;

		print("!!!! Sigmoid function either broke or not compatable");

		float loss=Float.MAX_VALUE;
		long time=System.currentTimeMillis();
		int iter=0;
		print("iterating...");
		while (loss >= 0.001f)
		{
			Base out =NDArray.dot(in, w1);
			out = NDArray.add(out, b1);
			// out = new Sigmoid().forward(out);
			out = NDArray.dot(out, w2);
			out = NDArray.add(out, b2);
			// out = new Sigmoid().forward(out);

			output = out;
			out = new MSE().forward(out, tr);

			loss = out.get(0);
			// print("loss :" + loss);

			out.setGrad(1);
			out.backward();

			opt.step();
			opt.zeroGrad();
			iter++;
			// Thread.sleep(100);
		}
		time = System.currentTimeMillis() - time; // ≈ 80768 millis.
		print(time + " millis to complate " + iter + " iterations");
		print("final output");
		output.printArray();
		print(line(30));
	}
	void trainMAE(Optimizer opt, Base w1, Base w2, Base b1, Base b2, Base in, Base tr)
	{
		Base output=null;

		print("!!!! Sigmoid function either broke or not compatable");
		print("iterating...");
		float loss=Float.MAX_VALUE;
		long time=System.currentTimeMillis();
		int iter=0;
		while (loss >= 0.001f)
		{
			Base out =NDArray.dot(in, w1);
			out = NDArray.add(out, b1);
			// out = new Sigmoid().forward(out);
			out = NDArray.dot(out, w2);
			out = NDArray.add(out, b2);
			// out = new Sigmoid().forward(out);

			output = out;
			out = new MAE().forward(out, tr);

			loss = out.get(0);
			print("loss :" + loss);

			out.setGrad(1);
			out.backward();

			opt.step();
			opt.zeroGrad();
			iter++;
		}
		time = System.currentTimeMillis() - time; // ≈ 80768 millis.
		print(time + " millis to complate " + iter + " iterations");
		print("final output");
		output.printArray();
		print(line(30));
	}
	void trainBCE(Optimizer opt, Base w1, Base w2, Base b1, Base b2, Base in, Base tr)
	{
		Base output=null;

		print("!!!! Sigmoid function either broke or not compatable");
		print("BCE broke");
		print("iterating...");
		float loss=Float.MAX_VALUE;
		long time=System.currentTimeMillis();
		int iter=0;
		while (loss >= 0.001f)
		{
			Base out =NDArray.dot(in, w1);
			out = NDArray.add(out, b1);
			// out = new Sigmoid().forward(out);
			out = NDArray.dot(out, w2);
			out = NDArray.add(out, b2);
			out = new Sigmoid().forward(out);

			output = out;
			out = new BCE().forward(out, tr);

			loss = out.get(0);
			print("loss :" + loss);

			out.setGrad(1);
			out.backward();

			opt.step();
			opt.zeroGrad();
			iter++;
		}
		time = System.currentTimeMillis() - time; // ≈ 80768 millis.
		print(time + " millis to complate " + iter + " iterations");
		print("final output");
		output.printArray();
		print(line(30));
	}
	void trainMCCE(Optimizer opt, Base w1, Base w2, Base b1, Base b2, Base in, Base tr)
	{
		Base output=null;

		print("!!!! Sigmoid function either broke or not compatable");
		print("MCCE works but use absolute(Math.abs(x)) value");
		print("iterating...");
		float loss=Float.MAX_VALUE;
		long time=System.currentTimeMillis();
		int iter=0;
		while (Math.abs(loss) >= 0.001f)
		{
			Base out =NDArray.dot(in, w1);
			out = NDArray.add(out, b1);
			// out = new Sigmoid().forward(out);
			out = NDArray.dot(out, w2);
			out = NDArray.add(out, b2);
			// out = new Sigmoid().forward(out);

			output = out;
			out = new MCCE().forward(out, tr);

			loss = out.get(0);
			print("loss :" + loss);

			out.setGrad(1);
			out.backward();

			opt.step();
			opt.zeroGrad();
			iter++;
		}
		time = System.currentTimeMillis() - time; // ≈ 80768 millis.
		print(time + " millis to complate " + iter + " iterations");
		print("final output");
		output.printArray();
		print(line(30));
	}
	void test6()
	{
		print(decString("Test 6.0 approximation using optimizers.", "✓", 5));
		Base tr=NDArray.wrap(3, new int[]{5});
		Base w1=NDArray.rand(1, 1).setRequiresGradient(true);
		Base b=NDArray.ones(5).setRequiresGradient(true);

		LossFunc mse=new MSE();
		Optimizer gd=new GradientDescent(w1, b);
		// gd = new Adam(w1, b);
		// gd = new SGDM(w1, b);

		Base in=NDArray.wrap(new float[]{1,2,3,4,5}, new int[]{5});

		Base res=null;
		float loss=100;
		while (loss >= 0.001)
		{
			Base rs=NDArray.dot(w1, in);
			rs = NDArray.add(rs, b);
			res = rs;
			Base ls=mse.forward(rs, tr);
			loss = ls.get(0);
			print("loss :", loss);
			ls.setGrad(1);
			ls.backward();

			// print(loss);
			gd.step();

			gd.zeroGrad();

		}	
		print(line(10));
		System.out.print("result :");
		res.printArray();
	}
	void test5()
	{
		print(decString("Test 5.0 approximation using array class.", "✓", 5));
		// gradient descent example.
		float lr=0.0001f;

		Base t=NDArray.wrap(new float[]{ 12, 15}, 2);
		Base in=NDArray.wrap(new float[]{2,3,4}, 3);

		Base w=NDArray.rand(new int[]{3, 2}).setRequiresGradient(true);
		Base b=NDArray.wrap(1, new int[]{2}).setRequiresGradient(true);

		Base res=null;
		float loss=100;

		while (loss >= 0.0001f)
		{
			Base o=NDArray.add(NDArray.dot(in, w), b);

			res = o;

			o = NDArray.sub(t, o);
			o = NDArray.pow(o, 2);
			loss = o.get(0);
			print("loss :", loss);
			o.setGrad(1);
			o.backward();

			Base wgr=NDArray.mul(w.detachGradient(), lr);
			wgr = NDArray.sub(w, wgr);
			w.set(wgr);

			Base bgr=NDArray.mul(b.detachGradient(), lr);
			bgr = NDArray.sub(b, bgr);
			// NDArray bgr=b.sub(b.getGradient().mul(lr));
			b.set(bgr);

			w.zeroGrad();
			b.zeroGrad();
		}
		print(line(10));
		System.out.print("result :");
		res.printArray();
	}
	void test4() throws Exception
	{
		print(decString("Test 4.0 approximation using array class.", "✓", 5));

		float lr=0.001f;

		Base t=NDArray.wrap(12, new int[]{1});
		Base in=NDArray.wrap(2, new int[]{1});

		Base w=NDArray.wrap(0.5f, 1).setRequiresGradient(true);
		Base b=NDArray.ones(1).setRequiresGradient(true);

		Base res=null;
		float loss=100;
		while (loss > 0.001f)
		{
			Base o=NDArray.add(NDArray.dot(in, w), b);
			res = o;

			o = NDArray.sub(t, o);
			// print(o, o.gradientFunction);
			o = NDArray.pow(o, 2);
			loss = o.get(0);

			print("loss :" + loss);

			o.setGrad(1);
			o.backward();

			float wv=w.get(0) - w.getGrad(0) * lr;
			w.set(new int[]{0}, wv);

			float bv=b.get(0) - b.getGrad(0) * lr;
			b.set(new int[]{0}, bv);

			w.zeroGrad();
			b.zeroGrad();
		}
		print(line(10));
		print("result :", res.get(0), " ≈≈ ", t.get(0));
	}
	void test3() throws Exception
	{
		print(decString("Test 3.0 approximation using raw float", "✓", 5));

		float lr=0.001f; // learning rate.

		float t=12; // target.

		float in=2; // input.
		float w=0.5f; // weight
		float wg=0; // weight gradient.
		float b=1; // biase.
		float bg=0; // biase gradient.

		float loss=100;
		float result=0;
		while (loss >= 0.001) // of the loss less than 0.001, it is already approximated.
		{

			float r=w * in + b; // result ( prediction);
			result = r;
			// print(r);
			// calculating loss.
			float m=t - r; 
			loss = (float)Math.pow(m, 2); // loss.
			print(loss); // loss sholud be closer to 0.
			// calculating gradients.
			float g=2 * m;
			bg = g;
			wg = g * t;
			// print(wg + " , " + bg);
			w += wg * lr;
			b += bg * lr;
		}
		print("result :" + result + " ≈≈≈ " + t);
	}
	void test2()
	{
		System.out.println(decString("Test 2.0 XOR Test.", "=", 10));

		Base x=new Base(new float[]{0,1,0,0,1,0,1,1}, 4, 2);
		Base y=new Base(new float[]{1,0,1,0}, 4);

		int hiddenSize=5;
		// it works with hidden size starts from 2 upto ...

		Linear l1=new Linear(2, hiddenSize);
		Linear l2=new Linear(hiddenSize, 1);

		Activation a2=new Tanh();

		LossFunc lossFunc=new MSE();

		Optimizer optim=new Adam(l1.getParameters(), l2.getParameters());
		// sometimes when we use Adam optimizer it stuck to local minima, or unable to fit the dataset. so keep try again.
		// optim = new GradientDescent(l1.getParameters(), l2.getParameters());
		optim = new SGDM(l1.getParameters(), l2.getParameters());

		optim.learningRate = 0.01f;
		Base output=null;

		int ps=0;
		float lsv=1000;
		while (lsv >= 0.02f)
		{
			lsv = 0;
			for (int i=0;i < 4;i++)
			{
				Base X = l1.forward(x.slice(i));
				// X = a2.forward(X);
				X = l2.forward(X);
				// X = a2.forward(X);
				output = X;

				Base loss=lossFunc.forward(X, y.slice(i));
				lsv += loss.get(0);

				print("loss :", lsv);

				loss.setGrad(1);
				loss.backward();

				optim.step();
				optim.zeroGrad();
			}
			ps++;
		}
		print(output);
		output.printArray();
	}

	void test1()
	{
		System.out.println(decString("Test 1.0 layers layer backward pass.", "=", 10));

		Sequential sq=new Sequential();
		sq.add(new Conv1d(800, 1, 10, 21));
		// 800 - 21 + 1 = 780
		sq.add(new MaxPool1d(5));
		// 780 / 5 = 156
		sq.add(new Dropout(0.25f));
		// === 10,156.
		sq.add(new Conv1d(156, 10, 15, 7));
		// 156 - 7 + 1 = 150
		sq.add(new MaxPool1d(3));
		// 150 / 3 = 50
		sq.add(new Dropout(0.21f));
		// third  15,50
		sq.add(new Conv1d(50, 15, 30, 3));
		// 50 - 3 + 1 = 48
		sq.add(new MaxPool1d(3));
		// 48 / 3 = 16
		sq.add(new Dropout(0.15f));
		// 30,16
		sq.add(new Flatten());
		sq.add(new Linear(480, 100, true));
		sq.add(new Linear(100, 50, true));
		// 1,250

		// input.
		Base d=NDArray.rand(800);

		Base out=sq.forward(d);
		// System.out.println(decString("forward complete.", 10));
		out.setGrad(1);
		out.backward();
		System.out.println(decString("complete.", 10));
		// System.out.println(out);

	}
}
