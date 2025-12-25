import gss.*;
import gss.act.*;
import gss.arr.*;
import gss.layers.*;
import gss.lossfunctions.*;
import gss.optimizers.*;

import static gss.Util.*;
import static gss.arr.GradFunc.*;
import static gss.Functions.*;
import java.util.*;
import modules.*;

public class Test2_Func
{
	public static void test() throws Exception
	{

		new Test2_Func().a();
		System.out.println(line(50));

	}
	void a() throws Exception
	{

//		test1();
//		test2();
//		test3();
//		test4();
//		test5();
//		test61();
//		test62();
//		test63();
//		test7();
//		test8();
//		test9();
//		test10();
//		test11();
//		test12();
//		test13();
//		test14();
//		test15();
//		test16();
		test17();


	}
	void test17() throws InterruptedException
	{
		print(decString("Test 16. XOR module", "-", 7));

		Base in=new Base(new float[]{0,0,1,1,0,1,1,0}, 4, 2);
		Base target=new Base(new float[]{0,0,1,1});

		Module xor=new XOR();

		Optimizer opt=new Adam(xor.getParameters());

		LossFunc lossF=new MSE();

		println(opt.params);

		float loss=100;
		while (loss >= 0.1f)
		{
			Base X=xor.forward(in);

			X = lossF.forward(X, target);
			loss = X.get(0);

			println("loss", loss);

			Thread.sleep(1000);
		}

	}
	void test16()
	{
		print(decString("Test 16. sum gradient", "-", 7));

		Base b=NDArray.arange(30).reshape(2, 3, 5).setRequiresGradient(true);

		Base s1=NDArray.sum(b, 2);

		s1.setGrad(-5);

		Base sgd=s1.detachGradient();

		sgd.set(NDArray.arange(sgd.length).reshape(sgd.shape));

		Base igd=b.detachGradient();

		s1.backward();

		println(b, "sum===", s1, "grad", sgd, igd);

	}
	void test15()
	{
		print(decString("Test 15. more functions", "-", 7));

		Base b=NDArray.arange(20).reshape(2, 10);

		Base l=NDArray.log(b);

		Base s=NDArray.sqrt(b);

		Base e=NDArray.exp(b);

		Base m=NDArray.mean(b);

		Base n=NDArray.neg(b);

		Base i=NDArray.inv(b);

		Base b2=NDArray.mul(NDArray.rand(2, 10), 20);

		Base lt=NDArray.lt(b, b2);

		Base eq=NDArray.eq(b, b2);

		println("input 1", b, "log", l, "sqrt", s, "exp", e, "mean", m, "negate", n, "inv", i, "input 2", b2, "less than", lt, "equals", eq);

	}
	void test14()
	{
		print(decString("Test 14. slice test 2", "-", 7));

		Base a=NDArray.arange(30).reshape(3, 5, 2);

		Base b=a.slice(new int[][]{ar(1, -1),ar(0, -1, 2)});

		Base c=b.slice(ar(2));

		println(a, b, c);

	}
	void test13()
	{
		print(decString("Test 13. sum operations.", "-", 7));
		Base a=NDArray.arange(30).reshape(3, 5, 2).setRequiresGradient(true);

		Base b=NDArray.sum(a);

		println(a, b.gradientFunction, b);
		b = NDArray.sum(a, 0);
		println(b.gradientFunction, b);

		b.setGrad(3);
		b.backward();
	}
	void test12()
	{
		print(decString("Test 12. Basic math operations.", "-", 7));
		Base a=NDArray.wrap(5, 5, 2);
		Base b=NDArray.wrap(10, 5, 2).setRequiresGradient(true);

		println(a, b);

		Base c=NDArray.add(a, b);
		println("gradient :" + c.gradientFunction, c);

		print(getString("-", 20));

		c = NDArray.add(b, 20);
		println(c.gradientFunction, " +++ " + 20, c);

		print(getString("-", 20));

		c = NDArray.sub(a, b);
		println(c.gradientFunction, c);

		print(getString("-", 20));

		c = NDArray.sub(b, 3);
		println(c.gradientFunction, "--- " + 3, c);

		print(getString("-", 20));

		c = NDArray.sub(3, b);
		println(c.gradientFunction, "--- " + 3, c);

		print(getString("-", 20));

		c = NDArray.mul(a, b);
		println("gradient :" + c.gradientFunction, c);

		print(getString("-", 20));

		c = NDArray.mul(b, 20);
		println(c.gradientFunction, " +++ " + 20, c);

		print(getString("-", 20));

		c = NDArray.div(a, b);
		println(c.gradientFunction, c);

		print(getString("-", 20));

		c = NDArray.div(b, 3);
		println(c.gradientFunction, "--- " + 3, c);

		print(getString("-", 20));

		c = NDArray.div(3, b);
		println(c.gradientFunction, "--- " + 3, c);

		print(getString("-", 20));

		c = NDArray.pow(a, b);
		println(c.gradientFunction, c);

		print(getString("-", 20));

		c = NDArray.pow(b, 3);
		println(c.gradientFunction, "--- " + 3, c);

		print(getString("-", 20));

		c = NDArray.pow(3, b);
		println(c.gradientFunction, "--- " + 3, c);


	}
	void test11()
	{

		final Base b1=NDArray.arange(60).reshapeLocal(3, 4, 5);

		println(b1.shape, b1);

		final int axis=0;
		ArrayToFloatFunction cons=new ArrayToFloatFunction(){
			@Override
			public float apply(int[] p1)
			{
				float out=0;
				for (int i=0;i < b1.shape[axis];i++)
				{
					int[]ar=putAtIndex(p1, i, axis);
					out += b1.get(ar);
				}
				return out;
			}
		};
		loop(removeAtIndex(b1.shape, axis), cons);

	}
	void test10()
	{
		print(decString("Test 10.0 : array value test.", 9));
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
	public static void tree(Base arr, String t)
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
	public static void treeV(Value vl, String t)
	{
		System.out.println(t + vl);
		if (vl.args != null && vl.args.size() != 0)
			for (Value  vv:vl.args)
				treeV(vv, t.replace("_", " ").replace("|", " ") + "|_____ ");
	}
	void test9()
	{
		print(decString("Test 9.0 dot product test.", 8));
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
		Base bs=new Base(outData, out).setRequiresGradient(a.hasGradient() | b.hasGradient());
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
		print(s, bs.gradientFunction);
		if (bs.childs != null && bs.childs.size() != 0)
			for (Base b:bs.childs)
				draw(b, "   " + s);
	}
	void test8() throws InterruptedException
	{
		print(decString("Test 8.0 approximation test with different loss functions.", 7));

		int input=2;
		int output=6;
		Base w1=NDArray.rand(input, 5).setRequiresGradient(true);
		Base w2=NDArray.rand(5, output).setRequiresGradient(true);
		Base b1=NDArray.ones(5).setRequiresGradient(true);
		Base b2=NDArray.ones(output).setRequiresGradient(true);

		Base in=NDArray.wrap(new float[]{0.5f,0.2f}, input);
		Base tr=NDArray.wrap(new float[]{1,0,1,0,1,0}, 2, 3);

		Optimizer opt=new SGDM(w1, w2, b1, b2);
		// SGDM better with MSE, BCE
		// Adam MAE, MCCE
		// GradientDescent slow.

		float loss=0.01f;
		// iteration_time in millis.
		trainMSE(opt, w1, w2, b1, b2, in, tr); // ✓ ≈ 1275_980, 1225_952
		// trainMAE(opt, w1, w2, b1, b2, in, tr);  // ✓ ≈ 1939_1132, 1888_1016
		// trainBCE(opt, w1, w2, b1, b2, in, tr); // ✓ ≈ 3529_1746, 3335_1678, 3649_1689
		// trainMCCE(opt, w1, w2, b1, b2, in, tr); // ✓ ≈ 8804_3576, 8925_3913;

		System.out.println("completed!");

	}
	void trainMSE(Optimizer opt, Base w1, Base w2, Base b1, Base b2, Base in, Base tr)
	{
		Base output=null;

		float loss=Float.MAX_VALUE;
		long time=System.currentTimeMillis();
		int iter=0;
		print("iterating...");
		while (loss >= 0.01f)
		{
			Base out =NDArray.dot(in, w1);
			out = NDArray.add(out, b1);
			out = new Sigmoid().forward(out);
			out = NDArray.dot(out, w2);
			out = NDArray.add(out, b2);
			out = new Sigmoid().forward(out);

			output = out;
			out = new MSE().forward(out, tr);

			loss = out.get(0);
			// print("loss :" + loss);
			opt.zeroGrad();
			out.setGrad(1);
			out.backward();

			opt.step();
			iter++;
			// Thread.sleep(100);
		}
		time = System.currentTimeMillis() - time; // ≈ 80768 millis.
		print(time + " millis to complate " + iter + " iterations");
		print("final output");
		print("loss ", loss);
		print(output);
		print(line(30));
	}
	void trainMAE(Optimizer opt, Base w1, Base w2, Base b1, Base b2, Base in, Base tr)
	{
		Base output=null;

		print("iterating...");
		float loss=Float.MAX_VALUE;
		long time=System.currentTimeMillis();
		int iter=0;
		while (loss >= 0.01f)
		{
			Base out = NDArray.dot(in, w1);
			out = NDArray.add(out, b1);
			out = new Sigmoid().forward(out);
			out = NDArray.dot(out, w2);
			out = NDArray.add(out, b2);
			out = new Sigmoid().forward(out);

			output = out;
			out = new MAE().forward(out, tr);

			loss = out.get(0);
			// print("loss :" + loss);

			opt.zeroGrad();
			out.setGrad(1);
			out.backward();

			opt.step();
			iter++;
		}
		time = System.currentTimeMillis() - time; // ≈ 80768 millis.
		print(time + " millis to complate " + iter + " iterations");
		print("final output");
		print("loss ", loss);
		output.printArray();
		print(line(30));
	}
	void trainBCE(Optimizer opt, Base w1, Base w2, Base b1, Base b2, Base in, Base tr) throws InterruptedException
	{
		Base output=null;

		print("iterating...");
		float loss=Float.MAX_VALUE;
		long time=System.currentTimeMillis();
		int iter=0;
		while (loss >= 0.01f)
		{
			Base out =NDArray.dot(in, w1);
			out = NDArray.add(out, b1);
			out = new Sigmoid().forward(out);
			out = NDArray.dot(out, w2);
			out = NDArray.add(out, b2);
			out = new Sigmoid().forward(out);

			output = out;
			out = new BCE().forward(out, tr);

			loss = out.get(0);
			// print("loss :" + loss);

			out.setGrad(1);
			out.backward();

			opt.step();
			opt.zeroGrad();
			iter++;
			// Thread.sleep(500);
		}
		time = System.currentTimeMillis() - time; // ≈ 80768 millis.
		print(time + " millis to complate " + iter + " iterations");
		print("final output");
		print("loss ", loss);
		output.printArray();
		print(line(30));
	}
	void trainMCCE(Optimizer opt, Base w1, Base w2, Base b1, Base b2, Base in, Base tr)
	{
		Base output=null;
		// use Adam optimizer for fast iteration.

		print("iterating...");
		float loss=Float.MAX_VALUE;
		long time=System.currentTimeMillis();
		int iter=0;
		while (loss >= 0.1f)
		{
			Base out =NDArray.dot(in, w1);
			out = NDArray.add(out, b1);
			out = new Sigmoid().forward(out);
			out = NDArray.dot(out, w2);
			out = NDArray.add(out, b2);
			out = new Sigmoid().forward(out);

			output = out;
			out = new MCCE().forward(out, tr);

			loss = out.get(0);
			// print("loss :" + loss);

			opt.zeroGrad();
			out.setGrad(1);
			out.backward();

			opt.step();
			iter++;
		}
		time = System.currentTimeMillis() - time; // ≈ 80768 millis.
		print(time + " millis to complate " + iter + " iterations");
		print("final output");
		print("loss ", loss);
		output.printArray();
		print(line(30));
	}
	void test7()
	{
		print(decString("Test 7.0 approximation using optimizers.", "✓", 5));

		Base in=NDArray.rand(5);
		Base tr=new Base(new float[]{1,0,1,1,0});

		Linear l=new Linear(5, 5);

		LossFunc ls=new MSE();
		Optimizer gd=new SGDM(l.getParameters());

		Base res=null;
		float loss=100;
		while (loss >= 0.01f)
		{
			Base rs=l.forward(in);
			res = rs;
			rs = ls.forward(rs, tr);
			loss = rs.get1d(0);

			gd.zeroGrad();
			rs.setGrad(1f);
			rs.backward();

			gd.step();
			// print("loss ", loss);
		}
		print(getString("*", 30));
		print("loss ", loss);
		print(res);

	}
	void test63()
	{
		print(decString("Test 6.3.0 activation function test.", "✓", 5));

		Base inp=new Base(new float[]{-0.5f ,0 ,1 ,-5 ,3 ,5}).setRequiresGradient(true);
		print("input", inp);

		ac(inp, new Relu());
		ac(inp, new Sigmoid());
		ac(inp, new Softmax());
		ac(inp, new Tanh());
	}
	void ac(Base inp, Activation act)
	{
		print(getString("-", 10));
		Base rs=act.forward(inp);
		print(act, "result", rs);
		rs.setGrad(3);
		rs.backward();
		print("gardient", inp.detachGradient());

	}
	void test62()
	{
		print(decString("Test 6.2.0 loss function test.", "✓", 5));

		Base in=NDArray.rand(4).setRequiresGradient(true);
		Base tr=new Base(new float[]{1,0,1,1});

		print(in);
		print(tr);

		lossT(in.copy(), tr.copy(), new MSE()); // ✓
		lossT(in.copy(), tr.copy(), new MAE()); // ✓
		lossT(in.copy(), tr.copy(), new MCCE()); // ✓
		lossT(in.copy(), tr.copy(), new BCE()); // ✓

	}
	void lossT(Base in, Base tr, LossFunc ls)
	{
		print(ls);
		Optimizer o=new SGDM(in);
		float loss=100;
		int iter=0;
		while (Math.abs(loss) >= 0.001f)
		{
			Base lossV=ls.forward(in, tr);
			loss = lossV.get1d(0);
			// print(">> ", loss);

			o.zeroGrad();

			lossV.setGrad(1);
			lossV.backward();

			o.step();
			iter++;
		}
		print(iter, getString("-", 15));
		print("== loss :", loss);

	}
	void test61()
	{
		print(decString("Test 6.1.0 optimizers test.", "✓", 5));

		Base data=new Base(new float[]{10}).setRequiresGradient(true);
		data.set1dGrad(0, 1);
		print(data);
		print(decString("", 5));
		print(data.detachGradient());
		print(decString("", 5));

		Optimizer gd=new GradientDescent(data);

		int iter=0;
		while (data.get1d(0) >= 0.1f)
		{
			data.setGrad(1);
			gd.step();
			gd.zeroGrad();
			iter++;
		}
		print("GradientDescent reaches <= 0.1  in " + iter + " iterations");

		data.fill(10);
		data.set1dGrad(0, 1);

		gd = new Adam(data);

		iter = 0;
		while (data.get1d(0) >= 0.1f)
		{
			data.setGrad(1);
			gd.step();
			gd.zeroGrad();
			iter++;
		}
		print("Adam reaches <= 0.1  in " + iter + " iterations");

		data.fill(10);
		data.set1dGrad(0, 1);

		gd = new SGDM(data);

		iter = 0;
		while (data.get1d(0) >= 0.1f)
		{
			data.setGrad(1);
			gd.step();
			gd.zeroGrad();
			iter++;
		}
		print("SGDM reaches <= 0.1  in " + iter + " iterations");
		print(decString("done", 10));

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

		while (loss >= 0.01f)
		{
			Base o=NDArray.add(NDArray.dot(in, w), b);

			res = o;

			o = NDArray.sub(t, o);
			o = NDArray.pow(o, 2);
			loss = o.get(0);
			// print("loss :", loss);
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
		print("≈≈");
		t.printArray();
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

			// print("loss :" + loss);

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
			// print(loss); // loss sholud be closer to 0.
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
	void test2() throws InterruptedException
	{
		System.out.println(decString("Test 2.0 XOR Test.", "=", 10));
		Base x=new Base(new float[]{0,0,1,1,0,1,1,0}, 4, 2);
		Base y=new Base(new float[]{0,0,1,1});

		int hiddenSize=5;
		// it works with hidden size starts from 2 upto ...

		Linear l1=new Linear(2, hiddenSize);
		Linear l2=new Linear(hiddenSize, 1);

		Activation a2=new Tanh();
		LossFunc lossFunc=new MSE();
		/*
		 --- MSE
		 Adam >6000 iterations. Tanh(1200, )
		 GradientDescent >100_000 iterations. Tanh(27_700,25_800)
		 SGDM >50_000 iterations. Tanh(4600,3400)
		 --- BCE more accurate loss functiom
		 Adam >7000 iterations, Tanh(2100)
		 GradientDescent >100_000 iterations. Tanh(11300)
		 SGDM >30_000 iterations, Tanh(1400,2100)
		 --- MAE
		 Adam stuck. Tanh(3000,1800) sometimes.
		 GradientDescent stuck
		 SGDM stuck
		 --- MCCE accurate loss function.
		 Adam 10_000, Tanh(3800)
		 GradientDescent >100_000 stuck.Tanh(28000) iterations.
		 SGDM > 60_000. Tanh(3700)
		 */
		Optimizer optim=new Adam(l1.getParameters(), l2.getParameters()); // very fast. < 7000 iterations.
		// sometimes when we use Adam optimizer it stuck to local minima, or unable to fit the dataset. so keep try again.
		// optim = new GradientDescent(l1.getParameters(), l2.getParameters()); // very slow. >100,000 iterations.
		// optim = new SGDM(l1.getParameters(), l2.getParameters()); // very slow. > 49,000 iterations.

		Base output=null;

		int ps=0;
		float lsv=1000;
		while (lsv >= 0.05f)
		{
			Base X = l1.forward(x);
			X = a2.forward(X);
			X = l2.forward(X);
			X = a2.forward(X);
			output = X;

			Base loss=lossFunc.forward(X, y);
			lsv = loss.get(0);

			if (ps % 100 == 0)
				print(ps++, lsv);

			optim.zeroGrad();
			loss.setGrad(1);
			loss.backward();

			optim.step();
			ps++;
			// Thread.sleep(100);
		}
		print(decString("-", 30));
		print("total iterations ", ps);
		print(output);
	}
	void test1()
	{
		System.out.println(decString("Test 1.0 layers backward pass.", "=", 10));

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
