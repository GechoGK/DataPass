import gss.*;
import gss.act.*;
import gss.arr.*;
import gss.lossfunctions.*;

import static gss.Util.*;
import static gss.arr.NDArray.*;

public class Main
{
	public static void main(String[] args) throws Exception
	{

		new Main().a();

	}
	void a() throws Exception
	{
//		test1();
//		test2();
//		test3();
//		test4();
//		test5();
//		test6();
//
//		print(getString("=", 30));
//
//		Test1_NDArrayData.test();
//
//		print(getString("=", 30));
//
//		Test2_LayereTest.test();
//
//		print(getString("=", 30));
//
		Test2_Func.test();

		print(decString("Test completed.", "=", 10));

	}
	void test6() throws Exception
	{
		print(decString("Test 6.0 Value class for BCE", 10));
		Base b=new Base(new float[]{.2f,.5f,.9f}).setRequiresGradient(true);
		Base bt=new Base(new float[]{0,0,1}).setRequiresGradient(true);
		Base b2=b.copy();
		Base bt2=bt.copy();
		b.printArray();
		bt.printArray();

		Base c=new BCE().forward(b, bt);
		print(c);
		c.printArray();
		c.setGrad(1);
		c.backward();
		b.detachGradient().printArray();
		print(line(30));

		Base c2=TestValue.bce(b2, bt2);
		print(c2);
		c2.printArray();
		c2.setGrad(1);
		c2.backward();
		b2.detachGradient().printArray();

		// Test2_Func.tree(c2, "");

	}
	float[] convF(float[]a, float[]b)
	{
		// works.
		int len=(a.length + b.length) - 1;
		float[] f=new float[len];
		for (int i=0;i < len;i++) // increment by inc. default = 1.
		{
			float sm=0;
			int kr=b.length - 1;
			int ips=-(b.length - 1) + i;
			for (int k=0;k < b.length;k++)
			{
				int kp=ips + k;
				if (kp >= 0 && kp < a.length)
				{
					float kv=b[kr]; //  kernels.get(kr, chn, kp);
					sm += a[kp] * kv;
				}
				kr--;
			}
			f[i] = sm;
		}
		return f;
	}
	Base corr(Base a, Base b)
	{
		// works.
		int len=Math.max(a.length, b.length) - Math.min(a.length, b.length) + 1;
		Base f=new Base(len);
		int wi=0;
		int iinc=a.length > b.length ?1: 0;
		int kinc=b.length > a.length ?1: 0;
		int wr=b.length > a.length ?len - 1: 0;
		for (int w=0;w < len;w++)
		{
			float sm=0;
			int kl=0;
			for (int i=0;i < Math.min(a.length, b.length);i++)
			{
				float aa=b.get(i + wr);
				float bb=a.get(i + wi);
				sm += aa * bb;
				kl++;
			}
			wr -= kinc;
			wi += iinc;
			f.set(ar(w), sm);
		}
		return f;
	}
	Base conv(Base a, Base b)
	{
		// works.
		int len=Math.max(a.length, b.length) - Math.min(a.length, b.length) + 1;
		Base f=new Base(len);
		int wi=0;
		int iinc=a.length > b.length ?1: 0;
		int kinc=b.length > a.length ?1: 0;
		int wr=b.length > a.length ?len - 1: 0;
		for (int w=0;w < len;w++)
		{
			float sm=0;
			int kl=b.length - 1;
			for (int i=0;i < Math.min(a.length, b.length);i++)
			{
				float aa=b.get(kl - wr);
				float bb=a.get(i + wi);
				sm += aa * bb;
				kl--;
			}
			wr -= kinc;
			wi += iinc;
			f.set(ar(w), sm);
		}
		return f;
	}
	void test5()
	{
		print(decString("Test 5.0 convolution 1D valid.", 10));
		Base b1=NDArray.wrap(new float[]{1, 2, 3, 4, 5, 6}).setRequiresGradient(true);
		Base k1=NDArray.wrap(new float[]{2, 3, 4}).setRequiresGradient(true);

		Base b2=b1.copy();
		Base k2=k1.copy();

		Base o1=NDArray.convolve1d(b1, k1, 0);
		o1.setGrad(NDArray.arange(2, o1.length + 2));
		o1.backward();
		print(decString("input", 7));
		b1.printArray();
		print(line(7));
		b1.detachGradient().printArray();
		print(decString("kernel", 7));
		k1.printArray();
		print(line(7));
		k1.detachGradient().printArray();
		print(decString("output", 7));
		o1.printArray();
		print(line(7));
		o1.detachGradient().printArray();
		print(line(50));

		// Base b2=NDArray.wrap(new float[]{1, 2, 3, 4, 5, 6,7}).setRequiresGradient(true);
		// Base k2=NDArray.wrap(new float[]{2, 3, 4}).setRequiresGradient(true);

		Base o2=TestValue.convolve1d2(b2, k2, null);
		o2.setGrad(NDArray.arange(2, o2.length + 2));
		o2.backward();
		print(decString("input", 7));
		b2.printArray();
		print(line(7));
		b2.detachGradient().printArray();
		print(decString("kernel", 7));
		k2.printArray();
		print(line(7));
		k2.detachGradient().printArray();
		print(decString("output", 7));
		o2.printArray();
		print(line(7));
		o2.detachGradient().printArray();
		print(line(30));

		print(line(30));

		if (Util.equals(o1, o2))
			print(decString("data equals", "✓", 7));
		else
			throw new RuntimeException("faild to verify array data equality.");
		if (Util.equals(b1.detachGradient(), b2.detachGradient()) && Util.equals(k1.detachGradient(), k2.detachGradient()))
			print(decString("gradient equals", "✓", 7));
		else
			throw new RuntimeException("faild to verify array gradient equality.");
	}
	void test4()
	{
		// not tested with value type.
		// dot2 doesn't use Value class, so use Value class instead.
		print(decString("Test 4.0 dot product", 10));

		Base b1=NDArray.arange(20).reshapeLocal(2, 2, 5).setRequiresGradient(true);
		Base b2=NDArray.arange(50).reshapeLocal(2, 5, 5).setRequiresGradient(true);

		Base b3=NDArray.dot(b1, b2);
		// b1.printArray();
		// print(line(10));
		// b2.printArray();
		print(b3);
		b3.printArray();
		print(line(20));
		b3.setGrad(NDArray.arange(1, 41).reshapeLocal(b3.shape));
		b3.backward();
		b1.detachGradient().printArray();
		print(line(5));
		b2.detachGradient().printArray();
		print(line(30));

		Base d1=NDArray.arange(20).reshapeLocal(2, 2, 5).setRequiresGradient(true);
		Base d2=NDArray.arange(50).reshapeLocal(2, 5, 5).setRequiresGradient(true);

		Base d3=TestValue.dot2(d1, d2);
		// d1.printArray();
		// print(line(10));
		// d2.printArray();
		print(d3);
		d3.printArray();
		print(line(20));
		d3.setGrad(NDArray.arange(1, 41).reshapeLocal(d3.shape));
		d3.backward();
		d1.detachGradient().printArray();
		print(line(5));
		d2.detachGradient().printArray();
		print(line(30));

		if (Util.equals(b3, d3))
			print(decString("data equals", "✓", 7));
		else
			throw new RuntimeException("faild to verify array data equality.");
		if (Util.equals(b1.detachGradient(), d1.detachGradient()) && Util.equals(b2.detachGradient(), d2.detachGradient()))
			print(decString("gradient equals", "✓", 7));
		else
			throw new RuntimeException("faild to verify array gradient equality.");
	}
	void test3()
	{
		print(decString("Test 3.0 sigmoid test", 10));
		Base bg=NDArray.arange(20).reshapeLocal(2, 10).setRequiresGradient(true);

		Base o=new Tanh().forward(bg);
		o.printArray();
		o.setGrad(NDArray.arange(1, 21).reshapeLocal(2, 10));
		o.backward();
		print(line(10));
		bg.detachGradient().printArray();
		print(decString("compare", 7));

		Base b=NDArray.arange(20).reshapeLocal(2, 10).setRequiresGradient(true);

		Base b2=TestValue.tanh(b);
		print(b2);
		b2.printArray();
		print(line(10));
		b2.setGrad(NDArray.arange(1, 21).reshapeLocal(2, 10));
		b2.backward();
		b.detachGradient().printArray();

		if (Util.equals(o, b2))
			print(decString("data equals", "✓", 7));
		else
			throw new RuntimeException("faild to verify array equality.");
		if (Util.equals(bg.detachGradient(), b.detachGradient()))
			print(decString("gradient equals", "✓", 7));
		else
			throw new RuntimeException("faild to verify array equality.");
	}
	void test2()
	{
		// sigmoid gradient have problem.
		print(decString("Test 2.0 sigmoid test", 10));
		Base bg=NDArray.arange(20).reshapeLocal(2, 10).setRequiresGradient(true);

		Base o=new Sigmoid().forward(bg);
		o.printArray();
		o.setGrad(NDArray.arange(20).reshapeLocal(2, 10));
		o.backward();
		print(line(10));
		bg.detachGradient().printArray();
		print(decString("compare", 7));

		Base b=NDArray.arange(20).reshapeLocal(2, 10).setRequiresGradient(true);

		Base b2=TestValue.sigmoid(b);
		print(b2);
		b2.printArray();
		print(line(10));
		b2.setGrad(NDArray.arange(20).reshapeLocal(2, 10));
		b2.backward();
		b.detachGradient().printArray();

		if (Util.equals(o, b2))
			print(decString("data equals", "✓", 7));
		else
			throw new RuntimeException("faild to verify array equality.");
		if (Util.equals(bg.detachGradient(), b.detachGradient()))
			print(decString("gradient equals", "✓", 7));
		else
			print("faild to verify array gradient equality.");
	}
	void test1()
	{
		print(decString("Test 1.0 Relu test", 10));

		Base bg=NDArray.arange(20).reshapeLocal(2, 10).setRequiresGradient(true);

		Base o=new Relu().forward(bg);
		o.printArray();
		o.setGrad(NDArray.arange(1, 21).reshape(2, 10));
		o.backward();
		print(line(10));
		bg.detachGradient().printArray();
		print(decString("compare", 7));

		Base b=NDArray.arange(20).reshapeLocal(2, 10).setRequiresGradient(true);

		Base b2=TestValue.relu(b);
		print(b2);
		print(line(10));
		b2.setGrad(NDArray.arange(1, 21).reshape(2, 10));
		b2.backward();
		b.detachGradient().printArray();

		if (Util.equals(o, b2))
			print(decString("data equals", "✓", 7));
		else
			throw new RuntimeException("faild to verify array equality.");
		if (Util.equals(bg.detachGradient(), b.detachGradient()))
			print(decString("gradient equals", "✓", 7));
		else
			throw new RuntimeException("faild to verify array equality.");
	}
}
