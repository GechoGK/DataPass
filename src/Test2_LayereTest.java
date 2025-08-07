import gss.act.*;
import gss.arr.*;
import gss.layers.*;
import gss.lossfunctions.*;
import java.util.*;

import static gss.Util.*;
import static java.util.Arrays.*;

public class Test2_LayereTest
{
	public static void main2(String[] args)
	{

		new Test2_LayereTest().a();

		System.out.println(line(54));

	}
	void a()
	{


	}
	void test16()
	{
		System.out.println("========== Test 16.0 Conv1d layer backward pass. ==========");

		int[] sh={3,2,7};
		Base d1=NDArray.arange(length(sh)).reshapeLocal(sh).setRequiresGradient(true);

		Conv1d c=new Conv1d(7, 2, 4, 3); // (input_size, num_features, num_kernels, kernel_size).
		Base out=c.forward(d1);
		System.out.println(decString("backward pass", 7));
		out.fillGrad(1);
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
		o = NDArray.fullCorrelate1d(a, b);
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
		out.fillGrad(3);
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
		out.fillGrad(5);
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
		out.fillGrad(1);
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
		b.fillGrad(2);
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
		dot.fillGrad(1);
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

		d3.fillGrad(3);
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

		System.out.println(decString("Multi label binary cross entropy",  10));
		out = new MultiLabelBinaryCrossEntropy().forward(pred, trueLabel);
		System.out.println("error :" + out);
		out.printArray();
		System.out.println(line(30));

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
