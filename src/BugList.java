public class BugList
{
	/*
	 -------- Bugs --------
	 1.0 --Base.slice(int[]dim) and Base.slice(int[][]dim)
	 >> check length before creating a new array.
	 ✓ 1.1 -- NDArray.concatenate(Base b1,Base b2,int axis)
	 >> gradient function not implemented.
	 ✓ 1.2 -- reshapeLocal doesn't check for array length matches total length shape.
	 ✓ 1.3 -- slice doesn't transfer gradients.
	 >> it make the original array doesn't modify by the optimizers.
	 >> RNN depend on it.
	 >> LSTM depend in it.

	 */
	/*
	 -------- possible bugs to be expected --------
	 1.0 slice(...).slice(...);
	 1.1 reshapeLocal review.
	 1.2  fix directly assigning shapes, strides ans data.items.

	 */
}
