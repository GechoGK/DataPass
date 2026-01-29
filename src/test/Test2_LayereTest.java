package test;

import gss.*;
import gss.act.*;
import gss.arr.*;
import gss.layers.*;
import gss.lossfunctions.*;
import gss.optimizers.*;
import java.util.*;
import modules.*;

import static gss.arr.NDArray.*;
import static test.Test2_Func.*;
import static gss.Util.*;
import static java.util.Arrays.*;

public class Test2_LayereTest
{
	public static void test()
	{

		new Test2_LayereTest().a();

		System.out.println(line(54));

	}
	void a()
	{
		test1();
		test2();
		test3();
		test4();
		test5();
		test6();
		test7();
		test8();
		test9();
		test10();
		test11();
		test12();
		test13();
		test14();
		test15();
		test16();
		test17();
		test18();
		test19();
		test20();
		test21();
		test22();

	}
	void test22()
	{
		print(decString("Test 22. (Layer, Batch) normalization test.", "-", 7));
		float[][]dt={{3,5,2,8},{1,3,5,8},{3,2,7,9}};
		Base b=NDArray.wrap(dt).setRequiresGradient(true);

		LayerNorm ln=new LayerNorm(4);
		Base out=ln.forward(b);

		println(decString("layer normalization", 5), out);

		draw(out);
		// not tested.
		BatchNorm bn=new BatchNorm(3);
		Base out2=bn.forward(b);

		println(line(20), decString("batch normalization", 5), out2);

		draw(out2);

	}
	void test21()
	{
		// review embedding layer.
		print(decString("Test 21. Embedding layer stress test.", "-", 7));
		int voc_size=512;
		long t=System.currentTimeMillis();
		Embedding emb=new Embedding(voc_size, 127);
		t = System.currentTimeMillis() - t;
		print(t, " millis, embedded weights created");
		t = System.currentTimeMillis();
		float[]indf=new float[100];
		Random r=new Random();
		for (int i=0;i < indf.length;i++)
			indf[i] = r.nextInt(voc_size);

		Base ind=NDArray.wrap(indf, 100);
		t = System.currentTimeMillis() - t;
		print(t, " millis, indices prepared");

		t = System.currentTimeMillis();
		Base embOut1=emb.forwardWithIndices(ind);
		t = System.currentTimeMillis() - t;
		print(t, " millis, embedded layer forward  with indices complete! ", embOut1.shape);

		t = System.currentTimeMillis();
		Base onehot=NDArray.onehot(ind, emb.vocabSize);
		t = System.currentTimeMillis() - t;
		print(t, " millis, onehot vector created", onehot.shape);

		t = System.currentTimeMillis();
		Base embOut2 = emb.forward(onehot);
		t = System.currentTimeMillis() - t;
		print(t, " millis, forward with onehot or matmul(dot) completed! ", embOut2.shape);

		print("equals", Util.equals(embOut1, embOut2));

	}
	void test20()
	{
		print(decString("Test 20. simple Embedding layer test.", "-", 7));
		Embedding emb=new Embedding(5, 3);

		Base onehot=NDArray.wrap(new float[]{0,1,0,0,0,0,0,0,1,0}, 2, 5);
		Base embOut=emb.forward(onehot);

		println("one hot", onehot, "embedded", embOut);
		print(line(20));

		Base ind=NDArray.wrap(new float[]{1,3});
		Base embOut2=emb.forwardWithIndices(ind);

		println("indices", ind, "embedded with indices", embOut2);
		boolean eq=Util.equals(embOut, embOut2);
		print("embedded equality", eq, eq ?"✓✓✓✓✓✓✓✓✓✓✓✓": "XXXXXXXXXXXX");
		Base onehot2=NDArray.onehot(ind, 5);

		boolean eq2=Util.equals(onehot, onehot2);
		print("onehot equality", eq2, eq2 ?"✓✓✓✓✓✓✓✓✓✓": "XXXXXXXXXX");

	}
	void test19()
	{
		print(decString("Test 19. simple LSTM module test", "-", 7));
		LSTM lstm=new LSTM(3, 4);
		// sequence length > 1 in progress.(doesn't support for now).
		Base in = NDArray.arange(12).reshapeLocal(2, 2, 3); // NDArray.wrap(.1f, .2f, .3f, .5f, .9f, .7f).reshapeLocal(2, 3);
		Base p_hidd=NDArray.wrap(.4f, .5f, .6f, .7f).setRequiresGradient(true);
		Base p_cell=NDArray.wrap(.8f, .9f, 1.0f, 1.1f).setRequiresGradient(true);

		lstm.cellState = p_cell;
		lstm.hiddenState = p_hidd; 

		Base out=lstm.forward(in);
		println(line(20), out);
		print(line(20));

		// draw(out);

	}
	void test18()
	{
		print(decString("Test 18. simple RNN module next number prediction test.", "-", 7));
		Object[] dt=prepareData();
		Base trainX=NDArray.wrap((float[][])dt[0]);
		Base trainY=NDArray.wrap(asFloat((Float[])dt[1]));

		println(trainX, trainY);
		print(line(20));

		RNN rnn=new RNN(2, 1);
		LossFunc loss=new MSE();
		Adam opt=new Adam(rnn.getParameters());
		opt.learningRate = 0.0001f;

		print("learning rate", opt.learningRate);
		float lsv=100;
		int iter=0;
		while (lsv >= 0.0015f)
		{
			Base out=rnn.forward(trainX);
			Base ls=loss.forward(out, trainY);
			lsv = ls.get(0);

			if (iter++ % 1000 == 0)
				print("loss = " + lsv);

			opt.zeroGrad();
			ls.setGrad(1);
			ls.backward();
			opt.step();

		}
		Base test=rnn.forward(trainX);
		print(test.as1DArray(), trainY);
		print("matches =", isClose(trainY, test, 0.05f));
	}
	Object[]prepareData()
	{
		// some random seed.
		float[] data=rand(ar(6), -690715410).data.items;
		int seq_len=2;

		ArrayList<float[]>X=new ArrayList<>();
		ArrayList<Float>Y=new ArrayList<>();
		for (int i=0;i < data.length - seq_len;i++)
		{
			X.add(Arrays.copyOfRange(data, i, i + seq_len));
			Y.add(data[i + seq_len]);
		}
		return new Object[]{X.toArray(new float[0][]),Y.toArray(new Float[0])};
	}
	Module test17()
	{
		print(decString("Test 17. XOR module Test", "-", 7));

		Base in=new Base(new float[]{0,0,1,1,0,1,1,0}, 4, 2);
		Base target=new Base(new float[]{0,0,1,1});

		Module xor=new XOR();

		Optimizer opt=new Adam(xor.getParameters());

		LossFunc lossF=new BCE();

		println(line(30));

		float loss=100;
		int iter=0;
		while (loss >= 0.1f)
		{
			Base X=xor.forward(in);

			X = lossF.forward(X, target);
			loss = X.get(0);

			if (iter++ % 1000 == 0)
				print("loss", loss);

			opt.zeroGrad();
			X.setGrad(1);
			X.backward();
			opt.step();

		}
		print("final prediction", xor.forward(in));
		return xor;
	}
	void test16()
	{
		System.out.println("========== Test 16.0 Conv1d layer backward pass. ==========");

		int[] sh={3,2,7};
		Base d1=NDArray.arange(length(sh)).reshapeLocal(sh).setRequiresGradient(true);

		Conv1d c=new Conv1d(7, 2, 4, 3); // (input_size, num_features, num_kernels, kernel_size).
		Base out=c.forward(d1);
		System.out.println(decString("backward pass", 7));
		out.setGrad(1);
		out.backward();
		System.out.println("input " + out);
		out.detachGradient().printArray();
		System.out.println(decString("kernels ", 7));
		System.out.println(c.kernels);
		c.kernels.detachGradient().printArray();
		System.out.println(decString("biase ", 7));
		System.out.println(c.biase);
		c.biase.detachGradient().printArray();
		System.out.println(decString("input ", 7));
		System.out.println(d1);
		d1.detachGradient().printArray();

	}
	void test15()
	{
		System.out.println("========== Test 15.0 convolution and fullCorrelation test. ==========");

		Base a=NDArray.arange(1, 6);
		Base b=NDArray.arange(1, 4);

		a.printArray();
		b.printArray();
		System.out.println(line(10));
		Base o=NDArray.convolve1d(a, b);
		o.printArray();
		// 10, 16, 22
		System.out.println(line(30));
		a = NDArray.ones(3);
		a.printArray();
		b.printArray();
		System.out.println(line(10));
		o = NDArray.correlate1dFull(a, b, null);
		o.printArray();
		// 3,5,6,3,1

	}
	void test14()
	{
		System.out.println("========== Test 14.0 slice test. ==========");

		Base d1=NDArray.arange(60).reshapeLocal(6, 10);
		System.out.println(d1);
		d1.printArray();

		System.out.println(line(30));
		Base d2=d1.slice(1).transpose();
		System.out.println("off :" + d2.offset + ", shape :" + Arrays.toString(d2.shape) + ", strides :" + Arrays.toString(d2.strides));
		d2.printArray();

		System.out.println(line(30));

		Base c1=d2.slice(2);
		System.out.println("off :" + c1.offset + ", shape :" + Arrays.toString(c1.shape) + ", strides :" + Arrays.toString(c1.strides));
		c1.printArray();

		System.out.println(decString("get with single done.", 10));

		Base s1=d1.slice(new int[][]{r(2, 6, 2),r(5, 10)});
		System.out.println(s1);
		s1.printArray();

	}

	void test13()
	{
		System.out.println("========== Test 13.0 MaxPool1d layer backward pass. ==========");
		MaxPool1d mx=new MaxPool1d(3);

		Base d1=NDArray.arange(5 * 12).reshapeLocal(5, 12).setRequiresGradient(true);
		d1 = NDArray.mul(d1, NDArray.rand(5, 12)).setRequiresGradient(true);
		System.out.println(d1);
		d1.printArray();
		System.out.println(decString("after MaxPool1d layer forward.", 10));
		Base out=mx.forward(d1);
		System.out.println(out);
		out.printArray();
		System.out.println(decString("after MaxPool1d layer backward.", 10));
		out.setGrad(3);
		out.backward();
		Base g=d1.detachGradient();
		System.out.println(g);
		g.printArray();
		System.out.println(decString("MaxPool1d indexes", 10));
		for (int[] r:mx.index)
			System.out.println(Arrays.toString(r));

	}
	void test12()
	{
		System.out.println("========== Test 12.0 Dropout layer backward pass. ==========");
		Base d1=NDArray.arange(20).reshapeLocal(2, 10).setRequiresGradient(true);

		Dropout drp=new Dropout(0.5f);

		Base out=drp.forward(d1);
		Base grd=d1.detachGradient();

		grd.printArray();
		System.out.println(decString("after backward method", 10));
		out.setGrad(5);
		out.backward();
		grd.printArray();

	}
	void test11()
	{
		System.out.println("========== Test 11.0 Linear layer backward pass. ==========");
		Base d1=NDArray.arange(20).reshapeLocal(2, 10);

		Linear l1=new Linear(10, 5);

		Base out=l1.forward(d1);

		System.out.println(out);
		out.printArray();
		System.out.println(out.gradientFunction);
		out.setGrad(1);
		out.backward();
		l1.weight.detachGradient().printArray();


	}
	void test10()
	{
		System.out.println("========== Test 10.0 addition backward pass. ==========");

		Base a1=NDArray.arange(10).reshapeLocal(2, 5).setRequiresGradient(true);
		Base a2=NDArray.arange(10, 20).reshapeLocal(2, 5).setRequiresGradient(true);
		Base b= NDArray.add(a1, 100);

		System.out.println(b);
		System.out.println(b.gradientFunction);
		b.printArray();
		b.setGrad(2);
		b.backward();
		System.out.println(line(30));

		System.out.println(decString("grad a1", 10));
		a1.detachGradient().printArray();
		System.out.println(decString("grad a2", 10));
		a2.detachGradient().printArray();

	}
	void test9()
	{
		System.out.println("========== Test 9.0 dot product backward pass. ==========");

		Base d1=new Base(new float[]{0,1,2,3,4,5}, new int[]{1,3,2}).setRequiresGradient(true);
		Base d2=NDArray.arange(3 * 2 * 4).reshapeLocal(4, 2, 3).setRequiresGradient(true);
		// 1,3,4,3 shape
		// check 2==2
		// 

		// System.out.println(d1);
		// d1.printArray();
		// System.out.println(getString("-", 30));
		// System.out.println(d2);
		// d2.printArray();
		// System.out.println(decString("dot product", 5));
		Base dot=NDArray.dot(d1, d2);
		// System.out.println(dot);
		// dot.printArray();
		// System.out.println(dot.gradientFunction);
		dot.setGrad(1);
		dot.backward();

		System.out.println(decString("gradient of host", 10));
		System.out.println(dot);
		dot.detachGradient().printArray();
		System.out.println(decString("geadient of a", 10));
		d1 = d1.detachGradient();
		System.out.println(d1);
		d1.printArray();
		System.out.println(decString("gradient of b", 10));
		d2 = d2.detachGradient();
		System.out.println(d2);
		d2.printArray();



	}
	void test8()
	{
		System.out.println(decString("Test 8.0 Gradient function test.", "=", 10));
		System.out.println("=== tested on additionGradientTest");
		Base d1=NDArray.arange(20).reshapeLocal(2, 10).setRequiresGradient(true);
		Base d2=NDArray.arange(10).setRequiresGradient(true);

		Base d3=NDArray.add(d1, d2);

		System.out.println(d3);
		System.out.println(d3.gradientFunction);
		d3.printArray();
		System.out.println(line(30));

		d3.setGrad(3);
		d3.backward();

		d1.detachGradient().printArray();
		System.out.println(line(10));
		d2.detachGradient().printArray();
		d2.zeroGrad();
		System.out.println("after zero gradient");
		d2.detachGradient().printArray();
	}
	void test7()
	{
		System.out.println(decString("Test 7.0 loss functions test.", "=", 10));
		Base pred=NDArray.rand(10);
		Base trueLabel=NDArray.zeros(10);
		trueLabel.set(new int[]{3}, 1);
		System.out.println("predicted :" + pred);
		pred.printArray();
		System.out.println(line(10));
		System.out.println("true label :" + trueLabel);
		trueLabel.printArray();
		System.out.println(line(30));

		System.out.println(decString("BCE",  10));
		Base out=new BCE().forward(pred, trueLabel);
		System.out.println("error :" + out);
		out.printArray();
		System.out.println(line(30));

		System.out.println(decString("MAE",  10));
		out = new MAE().forward(pred, trueLabel);
		System.out.println("error :" + out);
		out.printArray();
		System.out.println(line(30));

		System.out.println(decString("MCCE",  10));
		out = new MCCE().forward(pred, trueLabel);
		System.out.println("error :" + out);
		out.printArray();
		System.out.println(line(30));

		System.out.println(decString("MSE",  10));
		out = new MSE().forward(pred, trueLabel);
		System.out.println("error :" + out);
		out.printArray();
		System.out.println(line(30));

//		System.out.println(decString("Multi label binary cross entropy",  10));
//		out = new MultiLabelBinaryCrossEntropy().forward(pred, trueLabel);
//		System.out.println("error :" + out);
//		out.printArray();
//		System.out.println(line(30));

	}
	void test6()
	{
		System.out.println(decString("Test 6.0 activation layers test.", "=", 10));
		Base in=NDArray.arange(-5, 5).reshape(2, 5);
		System.out.println(in);
		in.printArray();
		System.out.println(line(30));

		System.out.println(decString("sigmoid activation", 10));
		Base out=new Sigmoid().forward(in);
		System.out.println(out);
		out.printArray();
		System.out.println(line(30) + "\n");

		System.out.println(decString("tanh activation", 10));
		out = new Tanh().forward(in);
		System.out.println(out);
		out.printArray();
		System.out.println(line(30) + "\n");

		System.out.println(decString("relu activation", 10));
		out = new Relu().forward(in);
		System.out.println(out);
		out.printArray();
		System.out.println(line(30) + "\n");

		System.out.println(decString("softmax activation", 10));
		out = new Softmax().forward(in);
		System.out.println(out);
		out.printArray();


	}
	void test5()
	{
		System.out.println(decString("Test 5.0 Conv1d layer test.", "=", 10));
		Base d1=NDArray.arange(12 * 2).reshapeLocal(2, 12);
		System.out.println(d1);
		d1.printArray();
		System.out.println(line(30));

		Conv1d c1=new Conv1d(12, 2, 5, 3);
		Base out=c1.forward(d1);
		System.out.println(out);
		out.printArray();


	}
	void test4()
	{
		System.out.println(decString("Test 4.0 MaxPool1d module test.", "=", 10));

		Base d1=NDArray.arange(5 * 12).reshapeLocal(5, -1);
		System.out.println(d1);
		d1.printArray();
		System.out.println(line(30));

		MaxPool1d m1=new MaxPool1d(3);
		Base d2=m1.forward(d1);
		System.out.println(d2);
		d2.printArray();

	}
	void test3()
	{
		System.out.println(decString("Test 3.0 Dropout module test.", "=", 10));
		Base d1=NDArray.rand(3, 5);

		Dropout d=new Dropout(0.5f);
		System.out.println(d1);
		d1.printArray();
		System.out.println(line(30));
		Base d2=d.forward(d1);
		System.out.println(d2);
		d2.printArray();

	}
	public static void test2()
	{
		System.out.println(decString("Test 2.0 Sequential module test.", "=", 10));
		Base d1=NDArray.arange(10).reshape(2, 5);
		System.out.println("input: " + d1);
		Sequential sq=new Sequential();
		sq.add(
			new Linear(5, 7, true),
			new Linear(7, 10, true),
			new Linear(10, 25, true),
			new Linear(25, 7, true));

		Base out=sq.forward(d1);
		System.out.println("output: " + out);

	}
	public static void test1()
	{
		System.out.println(decString("Test 1.0 Linear module test.", "=", 10));
		System.out.println("input");
		Base d1=NDArray.arange(10).reshape(2, 5);
		System.out.println(d1);

		Linear l1=new Linear(5, 10, false);
		Base out=l1.forward(d1);
		System.out.println("output");
		System.out.println(out);
		line(35);
		System.out.println("linear weights and biases");
		System.out.println(l1.weight);
		System.out.println(l1.biase);

	}
}
