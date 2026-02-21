package test;

import gss.*;
import gss.DataLoader.*;
import gss.act.*;
import gss.arr.*;
import gss.layers.*;
import gss.lossfunctions.*;
import gss.optimizers.*;
import java.util.*;

import static gss.Util.*;
import static gss.arr.GradFunc.*;
import static gss.Functions.*;
import static gss.arr.NDArray.*;
import static gss.MathUtil.*;

public class Test2_Func
{
	public static void test() throws Exception
	{

		new Test2_Func().a();
		System.out.println(line(50));

		/*
		 ----- TO-DO -----
		 -- embedding 80 % review. -> use index method. ✓
		 -- LayerNorm (layer normalization) 50 %
		 -- BatchNorm (batch normalization) 50 %
		 -- conv2d module ✓  gradient ? // fix out shape ✓ use cache ✓
		 -- maxPool2d module ✓  gradient ? // fix out shape ✓ use cache ✓
		 -- avPool1d module ✓  gradient ? // fix out shape ✓ use cache ✓
		 -- avPool2d module ✓  gradient ? // fix out shape ✓ use cache ✓

		 >> dot product. review. use cache ✓
		 >> min ✓
		 >> max ✓
		 >> fix mean.
		 >> fix variance.
		 >> fix NDArray.wrap(.) -> Util.flatten(.) copied array.
		 !! fix functions to use map...
		 !! fix sumGradient -> since axis is an array, it is still using as an integer. ✓

		 +++ test againest value.
		 >> conv1d 		50%
		 >> conv2d .... 50%
		 >> maxpool1d 	✓
		 >> maxpool2d 	✓
		 >> avpool1d 	✓
		 >> avpool2d 	✓
		 >> RNN 		?
		 >> LSTM 		?
		 >> BatchNorm 	? XX
		 >> LayerNorm 	? XX
		 >> concat 		?
		 >> variance 	? XX
		 >> mean 		?
		 >> sum 		? XX
		 >> dot 		?

		 */

	}
	void a() throws Exception
	{

//		test1();
//		test2();
//		test3();
//		test41();
//		test42();
//		test43();
//		test5();
//		test6();
//		test7();
//		test8();
//		test9();
//		test10();
//		test11(); // fix it. ✓
//		test12();
//		test13();
//		test14(); // fix it. ✓
//		test15();
//		test16();
//		bug10();
//		bug11();
//		bug13();
//		test17();
//		test18();
//		test19();
//		test20();
//		test21();
//		test22();
//		test23();
//		test24();
//		test25();
//		test26();
//		test27();
//		test28();
		test29();

		/*
		 TO-DO
		 -- Random generators.
		 -- Trainer
		 -- export and import Arrays. ✓
		 -- import and export modules. ✓
		 -- Data supplier. maybe.
		 -- more modules. ...
		 -- Save model data and load.
		 -- conv layers padding and stride.
		 */

	}
	void test29()
	{

		// fix softmax
		// make Trainer.

	}
	void test28() throws Exception
	{
		print(decString("Test 30. Min,Mqx with axis test.", "-", 7));
		// activation functions doesn't support batch.
		Base b=NDArray.rand(2, 3, 10);

		Base mn1=NDArray.min(b);
		Base mn2=NDArray.min(b, true);
		Base mn3=NDArray.min(b, 2);
		Base mn4=NDArray.min(b, true, 2);

		Base mx1=NDArray.max(b);
		Base mx2=NDArray.max(b, true);
		Base mx3=NDArray.max(b, 1);
		Base mx4=NDArray.max(b, true, 1);

		println(b);

		println(mn1, mn2, mn3, mn4);
		print(line(20));
		println(mx1, mx2, mx3, mx4);

	}

	void inputB()
	{
		new Thread(new Runnable(){
				@Override
				public void run()
				{
					Scanner sc=new Scanner(System.in);
					String ln="";
					while ((ln = sc.nextLine()) != null && !ln.equals("exit"))
					{
						if (ln.startsWith("l"))
						{
							ln = ln.substring(1);
							float f=Float.valueOf(ln);
							// opt.learningRate = f;
							System.out.println("learning rate changed to :" + f);
						}
					}
				}
			}).start();
	}
	void test27() throws Exception
	{
		print(decString("Test 28.fromAxis and fromNonAxis test. ✓", "-", 7));
		/*
		 fromAxis
		 fromNonAxis
		 remove
		 */
		int[]sh={2,3,4,5,6};

		print(decString("From Axis", 7));

		int[]frmAx=fromAxis(sh, 0, 2);
		int[]frmAxD=fromAxis(sh, true, 0, 2);

		print(sh, "from axis", ar(0, 2), "=", frmAx);
		print(sh, "from axis(keepDim=true)", ar(0, 2), "=", frmAxD);

		assertEquals("length equality", length(frmAx) == length(frmAxD));
		assertEquals("array equality keepDim(false)", Arrays.equals(frmAx, ar(2, 4)));
		assertEquals("array equality keepDim(true)", Arrays.equals(frmAxD, ar(2, 1, 4, 1, 1)));

		print(decString("From Non Axis", 7));

		int[]frmNAx=fromNonAxis(sh, 0, 2);
		int[]frmNAxD=fromNonAxis(sh, true, 0, 2);

		print(sh, "from non axis", ar(0, 2), "=", frmNAx);
		print(sh, "from non axis(keepDim=true)", ar(0, 2), "=", frmNAxD);

		assertEquals("length equality", length(frmNAx) == length(frmNAxD));
		assertEquals("array equality keepDim(false)", Arrays.equals(frmNAx, ar(3, 5, 6)));
		assertEquals("array equality keepDim(true)", Arrays.equals(frmNAxD, ar(1, 3, 1, 5, 6)));

	}
	void test26()
	{
		print(decString("Test 26. embedding new dot product test.", "-", 7));
		Embedding e=new Embedding(50, 10);
		print("embedding weights generated.");

		Base indices=NDArray.wrap(asFloat(1, 2, 4, 20));
		Base oneHot=NDArray.onehot(indices, e.vocabSize);

		Base o=e.forward(oneHot);
		Base o2=e.forwardWithIndices(indices);
		// Base o2=o;

		println("one hot", o, " === ", o2);

		Base g=NDArray.arange(length(o.shape), o.shape);
		o.setGrad(g);
		o.backward();

		// println(g, e.embeddingWeight.detachGradient());

		println("both operation are equals = " + Util.equals(o, o2));

	}
	void test25()
	{
		print(decString("Test 25. new dot function test.", "-", 7));
		int sz=128;
		Base aa=NDArray.rand(sz, sz);
		Base bb=NDArray.rand(sz, sz);

		float[][]a=MathUtil.copy2(aa);
		float[][]b=MathUtil.copy2(bb);

		long t=System.currentTimeMillis();
		// Base out2=NDArray.dot(aa, bb); // 106000
		// float[][]out=MathUtil.dot(a, b); // 9000
		float[]out2=MathUtil.dot2(a, b);
		t = System.currentTimeMillis() - t;

		// print("result equals " + Util.equals(NDArray.wrap(out), NDArray.wrap(out2, sz, sz)));

		print(t + " millis to finish(" + sz + ", " + sz + ") arrays");
	}
	void test24()
	{
		print(decString("Test 24. conv2d gradient test.", "-", 7));
		Base i1=NDArray.arange(1, 19).reshapeLocal(1, 2, 3, 3).setRequiresGradient(true);
		Conv2d cnn=new Conv2d(3, 2, 2, 2, false);
		float[][][][]kr=
		{
			{
				{
					{4,3},
					{2,1}
				},
				{
					{8,7},
					{6,5}
				}
			},
			{
				{
					{12,11},
					{10,9}
				},
				{
					{16,15},
					{14,13}
				}
			}
		};
		Base krn=NDArray.wrap(kr).setRequiresGradient(true);
		cnn.setKernels(krn);

		Base out=cnn.forward(i1);

		float[][][][]outd={{
				{{356, 392},
					{464, 500}},
				{{836, 936},
					{1136, 1236}
				}}};
		Base res=NDArray.wrap(outd);

		println(decString("output", 10), out);

		out.fillGrad(2);
		out.backward();

		float[][][][]ing={
			{
				{
					{20, 44,  24},
					{48, 104, 56},
					{28, 60,  32}},
				{
					{36, 76,  40},
					{80, 168, 88},
					{44, 92,  48}
				}
			}};
		Base inpGrd=NDArray.wrap(ing);

		println(decString("input gradient result ", 20),  i1.detachGradient());
		println("result equals ::" + (Util.equals(out, res) ? "true : ✓": "false : X"));
		print("input gradient equals ::" + (Util.equals(inpGrd, i1.detachGradient()) ?"true : ✓": "false : X"));


	}
	void agrad(float[][]in, float[][]k, float[][]grd, float[][]aout, float[][]bout)
	{
		print("calculating input gradient");
		print(in.length + "," + in[0].length);
		print(k.length + "," + k[0].length);
		print(grd.length + "," + grd[0].length);
		for (int gr=0;gr < grd.length;gr++)
			for (int gc=0;gc < grd[0].length;gc++)
			{
				float gval=grd[gr][gc];
				for (int kr=0;kr < k.length;kr++)
					for (int kc=0;kc < k[0].length;kc++)
					{
						float kval=k[kr][kc];
						float ival=in[gr + kr][gc + kc];
						// ag += kval * gval;
						aout[gr + kr][gc + kc] += kval * gval;
						// kg += ival * gval;
						bout[kr][kc] += ival * gval;
					}
			}
		// calculate input gradient.
	}
	void test23()
	{
		print(decString("Test 23. maxPool2d, averagePool2d gradient test.", "-", 7));
		Base in=NDArray.arange(100).reshapeLocal(10, 10).setRequiresGradient(true);
		println(in);

		Base customGrd=NDArray.arange(1, 26).reshapeLocal(5, 5);
		print(getString("-", 20));

		// for maxpool1d gradient
		MaxPool2d mx1=new MaxPool2d(2);
		Base out=mx1.forward(in);
		out.setGrad(customGrd);
		out.backward();
		println("maxPool2d gradient", out.detachGradient(), in.detachGradient());
		print(getString("+", 20));

		in.zeroGrad();

		// for avpool1d gradient
		AvPool2d av1=new AvPool2d(2);
		out = av1.forward(in);
		out.setGrad(customGrd);
		out.backward();

		println("avPool1d gradient", out.detachGradient(), in.detachGradient());

	}
	void test22()
	{
		print(decString("Test 22. maxPool1d, averagePool1d gradient test.", "-", 7));
		Base in=NDArray.arange(100).reshapeLocal(5, 20).setRequiresGradient(true);
		println(in);

		Base customGrd=NDArray.arange(1, 21).reshapeLocal(5, 4);
		print(getString("-", 20));

		// for maxpool1d gradient
		MaxPool1d mx1=new MaxPool1d(5);
		Base out=mx1.forward(in);
		out.setGrad(customGrd);
		out.backward();
		println("maxPool1d gradient", out.detachGradient(), in.detachGradient());
		print(getString("+", 20));

		in.zeroGrad();

		// for avpool1d gradient
		AvPool1d av1=new AvPool1d(5);
		out = av1.forward(in);
		out.setGrad(customGrd);
		out.backward();

		println("avPool1d gradient", out.detachGradient(), in.detachGradient());

	}
	void test21()
	{
		print(decString("Test 21. maxPool2d, averagePool1d, averagePool2d functions test.", "-", 7));
		Base in=NDArray.arange(30);

		float[] mx=MathUtil.maxPool1d(in, 5);
		float[] av=MathUtil.averagePool1d(in, 5);

		Base in2=NDArray.arange(100).reshapeLocal(10, 10);

		float[][]mx2=MathUtil.maxPool2d(in2, 2, 2);
		float[][]av2=MathUtil.averagePool2d(in2, 2, 2);
		println(in, getString("-", 20), "maxpool result", mx, getString("-", 20), "average pool result", av);
		print(in2);
		print("mxpool2d result");
		println(mx2);
		print("avpool2d result");
		println(av2);
	}

	void test20()
	{
		print(decString("Test 20. conv2d function benchmark test. new method.", "-", 7));
		Base in=NDArray.arange(100 * 100).reshapeLocal(100, 100);
		Base k=NDArray.arange(30 * 30).reshapeLocal(30, 30);

		long t=0;
		float[][]out=new float[1][1];
		int cnt=5;
		long aTime=0,lTime=0,sTime=Long.MAX_VALUE;
		while (count(cnt))
		{
			// expect 25 get print messages.
			t = System.currentTimeMillis();
			out = MathUtil.conv2d(in, k);
			t = System.currentTimeMillis() - t;
			lTime = Math.max(lTime, t);
			sTime = Math.min(sTime, t);
			aTime += t;
			// print(out);
		}
		aTime = aTime / cnt;
		println("time :" + aTime + " millis in average(" + sTime + " - " + lTime + ")", "input", in.shape, "kernel", k.shape, "output", out.length);
		// println("output ndarray", NDArray.wrap(out), "output float", NDArray.wrap(out2));
		resetCount();


	}
	void test19()
	{
		print(decString("Test 19. conv1d benchmark test. new method.", "-", 7));
		Base in=NDArray.rand(100);
		Base k=NDArray.rand(35);

		long t=0;
		Base out=NDArray.zeros(1);
		float[]out2=new float[1];
		int cnt=100;
		float aTime=0,lTime=0,sTime=Float.MAX_VALUE;
		while (count(cnt))
		{
			t = System.nanoTime();
			// out = NDArray.convolve1d(in, k);
			out2 = MathUtil.conv1d(in, k);
			t = System.nanoTime() - t;
			lTime = Math.max(lTime, t);
			sTime = Math.min(sTime, t);
			aTime += t;
			// print(ff);
		}
		aTime = aTime / cnt;
		// println(in, k);
		println("time :" + aTime + " millis in average(" + sTime + " - " + lTime + ")", "input", in.shape, "kernel", k.shape, "output", out.shape, "float shape :" + out2.length);
		print("equals", Util.isClose(NDArray.wrap(out2), out));
		// println("output ndarray", out, "output float", out2);
		resetCount();

	}
	public static void resetCount()
	{
		count = 0;
	}
	static int count=0;
	public static boolean count(int lim)
	{
		if (count++ < lim)
			return true;
		return false;
	}
	void test18()
	{
		print(decString("Test 18. variance test.", "-", 7));
		float[][]dt={{3,5,2,8},{1,3,5,8},{3,2,7,9}};
		Base b=NDArray.wrap(dt);

		print("orig", b);

		Base var=NDArray.variance(b, 1);

		print("variance", var);


	}
	void test17()
	{
		print(decString("Test 17. NDArray stress test.", "-", 7));
		float[] f=new float[1024 * 1024];
		long t=0;
		float s=0;
		t = System.currentTimeMillis();
		for (int i=0;i < f.length;i++)
			s = f[i] * 2;
		t = System.currentTimeMillis() - t;
		print(t, "millis, raw float");

		print(line(20));
		Base b=NDArray.wrap(f);
		t = System.currentTimeMillis();
		for (int i=0;i < f.length;i++)
			s = b.get(i) * 2;
		t = System.currentTimeMillis() - t;
		print(t, "millis, NDArray(1d).get(...)");

		print(line(20));
		t = System.currentTimeMillis();
		for (int i=0;i < f.length;i++)
			s = b.get1d(i) * 2;
		t = System.currentTimeMillis() - t;
		print(t, "millis, NDArray(1d).get1d(...)");

		print(line(20));
		t = System.currentTimeMillis();
		for (int i=0;i < f.length;i++)
			s = b.getRaw(i) * 2;
		t = System.currentTimeMillis() - t;
		print(t, "millis, getRaw(...)");

		print(line(20));
		t = System.currentTimeMillis();
		for (int i=0;i < f.length;i++)
			s = get(b, i) * 2;
		t = System.currentTimeMillis() - t;
		print(t, "millis, getRaw(...)");

		b = b.transpose();
		print(line(20));
		t = System.currentTimeMillis();
		for (int i=0;i < f.length;i++)
			s = b.get(i) * 2;
		t = System.currentTimeMillis() - t;
		print(t, "millis, NDArray(1d).T.get(...)");

		// b = b.transpose(); // already transposed.
		print(line(20));
		t = System.currentTimeMillis();
		for (int i=0;i < f.length;i++)
			s = b.get1d(i) * 2;
		t = System.currentTimeMillis() - t;
		print(t, "millis, NDArray(1d).T.get1d(...)");

	}
	float get(Base b, int i)
	{
		return b.data.items[i];
	}

	void bug13()
	{
		Base bo1=NDArray.arange(20, ar(2, 4, 5)).setRequiresGradient(true);
		Base bo2=NDArray.wrap(3, bo1.shape).setRequiresGradient(true);

		Base bo=NDArray.add(bo1, bo2);
		print(bo);

		Base b=bo.slice(new int[][]{r(0, 1)});
		print("==original has grad:", bo.hasGradient(), "sliced has grad:", b.hasGradient());
		print("==original has grad func:", bo.gradientFunction, "\nsliced has grad func:", b.gradientFunction);
		Base b2=NDArray.wrap(5, b.shape).setRequiresGradient(true);

		Base o=NDArray.add(b, b2);
		print(o);
		print(line(20));

		draw(o);

		o.setGrad(7);
		o.backward();

		println(b2.detachGradient(), bo1.detachGradient(), bo2.detachGradient());

	}
	void bug11()
	{
		// print(decString("bug 11. concat backward gradient test", "-", 7));
		Base b1=NDArray.arange(20).reshapeLocal(2, 2, 5).setRequiresGradient(true);
		Base b2=NDArray.arange(12).reshapeLocal(2, 6).setRequiresGradient(true);

		Base comb=NDArray.concat(b1, b2, 1);

		println("concatenation reult", comb);
		print(line(20));

		comb.setGrad(NDArray.wrap(2, comb.shape));
		print(comb.detachGradient());
		comb.backward();

		println("child grads", b1.detachGradient(), b2.detachGradient());

	}
	void bug10()
	{
		Base bo=NDArray.arange(24, ar(2, 3, 4));

		Base b=bo.slice(5, 0); //5,5 should throw error.
		// -------------^ equivalent to bo.slice(1);
		// this mistake is occured because of "shapeToIndex" lazy broadcasting....
		Base b2=bo.slice(new int[][]{r(1)}); // should throw range error.
		// b2.shape must not equal to [3,3,4]
		println(bo, b, b2);

	}

	void test16()
	{
		print(decString("Test 16. concatinate test", "-", 7));

		Base b=NDArray.arange(10).reshape(2, 5);
		print(">>", b.get(20, 3));

		int[] tsh=copyB(ar(1, 2, 3), 5);
		print(tsh, assertEquals(Arrays.equals(tsh, ar(0, 0, 1, 2, 3))));

		tsh = copyB(ar(9, 4, 1, 2, 7), 3);
		print(tsh, assertEquals(Arrays.equals(tsh, ar(1, 2, 7))));

		print("concateates shape", assertEquals(Arrays.equals(concatShape(ar(1, 2, 3), ar(1, 5, 3), 1), ar(1, 7, 3))));

		print("equals except", equalsExcept(ar(1, 2, 3), ar(1, 5, 3), 1));

		print("map test");
		Base mapped=NDArray.map(ar(2, 3), new ArrayFunction(){
				@Override
				public float apply(int[] p1, Object[]o)
				{
					return 5;
				}
			});
		print("mapped test", assertEquals(Util.equals(mapped, NDArray.wrap(asFloat(5, 5, 5, 5, 5, 5), 2, 3))));

		println(line(20), "concat test");
		Base b1=NDArray.arange(9).reshapeLocal(3, 3);
		Base b2=NDArray.arange(10, 25).reshapeLocal(3, 5);

		Base combined=NDArray.concat(b1, b2, 1);
		println(b1, b2, combined);
		print("concatenation test 1 ", assertEquals(Util.equals(combined, NDArray.wrap(asFloat(0, 1, 2, 10, 11, 12, 13, 14,
																							   3, 4, 5, 15, 16, 17, 18, 19,
																							   6, 7, 8, 20, 21, 22, 23, 24), 3, 8))));

		b1 = NDArray.wrap(asFloat(1, 6, 4));
		b2 = NDArray.wrap(asFloat(3, 8, 6, 0, 9));
		Base comb2=NDArray.concat(b1, b2, 0);

		println(b1, b2, comb2);
		print("concatenation test 2", assertEquals(Util.equals(comb2, NDArray.wrap(asFloat(1, 6, 4, 3, 8, 6, 0, 9)))));

	}
	void test15() throws Exception
	{
		print(decString("Test 15. Import and Export Arrays", "-", 7));
		// test export and import array.

		Base b1=NDArray.rand(2, 4, 10).setRequiresGradient(true);

		String path="/sdcard/testArray.ndbin";
		NDIO.save(b1, path, NDIO.FileType.BINARY);
		Base b2=NDIO.load(path, NDIO.FileType.BINARY);
		boolean result=Util.equals(b1, b2, true);
		print("array from binary load and save result ", result);

		path = "/sdcard/testArray.json";
		NDIO.save(b1, path, NDIO.FileType.JSON);
		b2 = NDIO.load(path, NDIO.FileType.JSON);
		result = Util.equals(b1, b2, true);
		print("array from json load and save result ", result);

		path = "/sdcard/testArray.txt";
		NDIO.save(b1, path, NDIO.FileType.TEXT);
		b2 = NDIO.load(path, NDIO.FileType.TEXT);
		result = Util.equals(b1, b2, true);
		print("array from text load and save result ", result);

	}
	void test14()
	{
		print(decString("Test 14. sum gradient", "-", 7));

		Base b=NDArray.arange(30).reshape(2, 3, 5).setRequiresGradient(true);

		Base s1=NDArray.sum(b, 2);

		s1.setGrad(-5);

		Base sgd=s1.detachGradient();

		sgd.set(NDArray.arange(sgd.length).reshape(sgd.shape));

		Base igd=b.detachGradient();

		s1.backward();

		println(b, "sum===", s1, "grad", sgd, igd);

	}
	void test13()
	{
		print(decString("Test 13. more functions", "-", 7));

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
	void test12()
	{
		print(decString("Test 12. slice test 2", "-", 7));

		Base a=NDArray.arange(30).reshape(3, 5, 2);

		Base b=a.slice(new int[][]{ar(1, -1),ar(0, -1, 2)});

		Base c=b.slice(ar(2));

		println(a, b, c);

	}
	void test11()
	{
		print(decString("Test 11. sum operations.", "-", 7));
		Base a=NDArray.arange(30).reshape(3, 5, 2).setRequiresGradient(true);

		Base b=NDArray.sum(a);

		println(a, b.gradientFunction, b);
		b = NDArray.sum(a, 0);
		println(b.gradientFunction, b);

		b.setGrad(3);
		a.zeroGrad();
		b.backward();
		print(decString("gradients", 5));
		println(b.detachGradient(), a.detachGradient());
	}
	void test10()
	{
		print(decString("Test 10. Basic math operations.", "-", 7));
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
	void test9()
	{

		final Base b1=NDArray.arange(60).reshapeLocal(3, 4, 5);

		println(b1.shape, b1);

		final int axis=0;
		ArrayFunction cons=new ArrayFunction(){
			@Override
			public float apply(int[] p1, Object[]o)
			{
				float out=0;
				for (int i=0;i < b1.shape[axis];i++)
				{
					int[]ar=insert(p1, i, axis);
					out += b1.get(ar);
				}
				return out;
			}
		};
		loop(remove(b1.shape, axis), cons);

	}
	void test8()
	{
		print(decString("Test 8.0 : array value test.", 9));
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
	void test7()
	{
		print(decString("Test 7.0 dot product test.", 8));
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
	public static void draw(Base b)
	{
		draw(b, "");
	}
	public static void draw(Base bs, String s)
	{
		print(s, bs.gradientFunction);
		if (bs.childs != null && bs.childs.size() != 0)
			for (Base b:bs.childs)
				draw(b, "   " + s);
	}
	public static void draw2(Base bs)
	{
		print(bs.gradientFunction);
		if (bs.childs != null && bs.childs.size() != 0)
		{
			int pos=0;
			for (Base b:bs.childs)
				print("  " + pos++   + ". " + b.gradientFunction);
			print("      type number to explore, -1 to exit.");
			int ps=input("#").nextInt();
			if (ps < 0)
				return;
			if (ps >= bs.childs.size())
			{
				print("      gradient not found at: " + ps + ".");
				draw2(bs);
			}
			else
				draw2(bs.childs.get(ps));
		}
	}
	void test6() throws InterruptedException
	{
		print(decString("Test 6.0 approximation test with different loss functions.", 7));

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
	void test5()
	{
		print(decString("Test 5.0 approximation using optimizers.", "✓", 5));

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
	void test43()
	{
		print(decString("Test 4.3.0 activation function test.", "✓", 5));

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
	void test42()
	{
		print(decString("Test 4.2.0 loss function test.", "✓", 5));

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
	void test41()
	{
		print(decString("Test 4.1.0 optimizers test.", "✓", 5));

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
	void test3()
	{
		print(decString("Test 3.0 approximation using array class.", "✓", 5));
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
	void test2() throws Exception
	{
		print(decString("Test 2.0 approximation using array class.", "✓", 5));

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
	void test1() throws Exception
	{
		print(decString("Test 1.0 approximation using raw float", "✓", 5));

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
}
