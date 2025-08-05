import gss.arr.*;
import gss.layers.*;

import static gss.Util.*;

public class Test2_Model
{
	void test1()
	{
		System.out.println(decString("Test 1.0 Conv1d layer backward pass.", "=", 10));

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
		out.fillGrad(1);
		out.backward();
		System.out.println(decString("complete.", 10));
		// System.out.println(out);

	}
}
